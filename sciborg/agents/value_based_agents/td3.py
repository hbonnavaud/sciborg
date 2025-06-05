from copy import deepcopy
from typing import Union, Type
import numpy as np
import torch
from torch import optim, nn
from gym.spaces import Box

from ..utils import ReplayBuffer, NeuralNetwork
from .value_based_agent import ValueBasedAgent


class TD3(ValueBasedAgent):
    name = "TD3"

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma: float = 0.99,
                 exploration_noise_std: float = 0.1,
                 target_action_noise_std: float = 0.2,
                 target_action_max_noise: float = 0.5,
                 policy_update_frequency: int = 2,
                 batch_size: int = 256,
                 replay_buffer_size: int = int(1e6),
                 steps_before_learning: int = int(25e3),

                 layer1_size: int = 256,
                 layer2_size: int = 256,

                 tau: Union[None, float] = None,
                 actor_tau: float = 0.001,
                 critic_tau: float = 0.001,

                 learning_rate: Union[None, float] = None,
                 actor_lr: float = 0.000025,
                 critic_lr: float = 0.00025,

                 # N.B.: Type[torch.optim.Optimizer] mean that the argument should be a subclass of optim.Optimizer
                 optimizer: Union[None, Type[torch.optim.Optimizer]] = None,
                 actor_optimizer: Type[torch.optim.Optimizer] = optim.Adam,
                 critic_optimizer: Type[torch.optim.Optimizer] = optim.Adam,
                 ):
        """
        Args:
            observation_space: Agent's observations space.
            action_space: Agent's actions space.
            gamma: Value of gamma in the critic's target computation formulae.
            layer1_size: Size of actor and critic first hidden layer.
            layer2_size: Size of actor and critic second hidden layer.
            tau: Tau for target critic and actor convergence to their non-target equivalent. If set, this value
                overwrite both 'actor_tau' and 'critic_tau' hyperparameters.
            actor_tau:
            critic_tau:
            learning_rate: Learning rate of actor and critic modules. If set, this value overwrite both
                'actor_lr' and 'critic_lr' hyperparameters.
            actor_lr: Actor learning rate. Overwritten by 'learning_rate' if it is set.
            critic_lr: Critic learning rate. Overwritten by 'learning_rate' if it is set.
            optimizer: Actor and Critic optimizer. Must be an instance of torch.optim.Optimizer. If set, this value
                overwrite both 'actor_optimizer' and 'critic_optimizer' hyperparameters.
            actor_optimizer: Actor optimizer. Must be an instance of torch.optim.Optimizer. Overwritten by optimizer
                if set.
            critic_optimizer: Critic optimizer. Must be an instance of torch.optim.Optimizer. Overwritten by optimizer
                if set.
        """

        super().__init__(observation_space, action_space)

        self.gamma = gamma
        self.steps_before_learning = steps_before_learning
        self.layer_1_size = layer1_size
        self.layer_2_size = layer2_size
        self.exploration_noise_std = exploration_noise_std
        self.target_action_noise_std = target_action_noise_std
        self.target_action_max_noise = target_action_max_noise
        self.policy_update_frequency = policy_update_frequency
        self.learning_steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.batch_size = batch_size

        # Setup critic and its target
        self.critic_1 = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, 1), nn.Tanh())
        self.target_critic_1 = deepcopy(self.critic_1)

        self.critic_2 = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, 1), nn.Tanh())
        self.target_critic_2 = deepcopy(self.critic_2)

        optimizer = critic_optimizer if optimizer is None else optimizer
        assert issubclass(optimizer, optim.Optimizer), "The optimizer should be a subclass of torch.optim.Optimizer"
        self.critic_optimizer = optimizer(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                          lr=critic_lr if learning_rate is None else learning_rate)
        self.critic_tau = critic_tau if tau is None else tau

        # Setup actor and its target
        self.actor = NeuralNetwork(
            module=nn.Sequential(
                nn.Linear(self.observation_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, self.action_size), nn.Tanh()),
            device=self.device,
            tau=actor_tau if tau is None else tau,
            learning_rate=actor_lr if learning_rate is None else learning_rate,
            optimizer_class=actor_optimizer if optimizer is None else optimizer
        )
        self.target_actor = deepcopy(self.actor)

        self.action_noise = torch.distributions.normal.Normal(
            torch.zeros(self.action_size), torch.full((self.action_size,), self.exploration_noise_std))

    def action(self, observation, explore=True):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor(observation).to(self.device)
            if not self.under_test and explore:
                action += self.action_noise.sample()
            action = action.cpu().detach().numpy()

            # Fit action to our action_space
            action = self.scale_action(action, Box(-1, 1, (self.action_size,)))
        return action

    def learn_interaction(self, *interaction_data):
        assert not self.under_test
        self.replay_buffer.append(interaction_data)

    def get_value(self, observations, actions=None):
        with torch.no_grad():
            if actions is None:
                actions = self.actor(observations)
            if isinstance(observations, np.ndarray):
                observations = torch.Tensor(observations)
            if isinstance(actions, np.ndarray):
                actions = torch.Tensor(actions)
            critic_value = self.critic_1(torch.concat((observations, actions), dim=-1))
        return critic_value.flatten().detach().numpy()

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        if learn and not self.under_test:
            self.replay_buffer.append((self.last_observation, action, reward, new_observation, done))
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def learn(self):
        assert not self.under_test
        if self.train_interactions_done >= self.steps_before_learning and len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, new_states, dones = self.replay_buffer.sample(self.batch_size)

            with (torch.no_grad()):
                clipped_noise = torch.randn_like(actions, device=self.device) * self.target_action_noise_std
                clipped_noise = torch.clamp(clipped_noise, -self.target_action_max_noise, self.target_action_max_noise)
                target_actions = self.target_actor(new_states) + clipped_noise
                target_actions = self.scale_action(target_actions, Box(-1, 1, (self.action_size,)))
                target_critic_value_1 = self.target_critic_1(torch.concat((new_states, target_actions), dim=-1))
                target_critic_value_2 = self.target_critic_2(torch.concat((new_states, target_actions), dim=-1))
                target_critic_value = torch.min(target_critic_value_1, target_critic_value_2)

            target = (rewards + self.gamma * (1 - dones) * target_critic_value.squeeze()).view(self.batch_size, 1)
            critic_1_value = self.critic_1(torch.concat((states, actions), dim=-1))
            critic_2_value = self.critic_2(torch.concat((states, actions), dim=-1))

            critic_1_loss = torch.nn.functional.mse_loss(target, critic_1_value)
            critic_2_loss = torch.nn.functional.mse_loss(target, critic_2_value)
            critic_loss = critic_1_loss + critic_2_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.train_interactions_done % self.policy_update_frequency == 0:
                actions = self.actor(states)
                actions = self.scale_action(actions, Box(-1, 1, (self.action_size,)))
                actor_loss = - self.critic_1(torch.concat((states, actions), dim=-1))
                actor_loss = torch.mean(actor_loss)
                self.actor.learn(actor_loss)

                self.target_actor.converge_to(self.actor)
                for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                    target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                    target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
        self.learning_steps_done += 1