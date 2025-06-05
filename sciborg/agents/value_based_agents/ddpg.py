import warnings
from copy import deepcopy
from typing import Union, Type
import numpy as np
import torch
from torch import optim, nn
from gym.spaces import Box
from ..utils import ReplayBuffer
from .value_based_agent import ValueBasedAgent
from ..utils.copy_weights import copy_weights
import inspect


class DDPG(ValueBasedAgent):
    name = "DDPG"

    def __init__(self,
                 observation_space,
                 action_space,
                 gamma: float = 0.99,
                 exploration_noise_std: float = 0.1,
                 policy_update_frequency: int = 2,
                 batch_size: int = 256,
                 replay_buffer_size: int = int(1e6),
                 steps_before_learning: int = 1000,

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
                 critic_optimizer: Type[torch.optim.Optimizer] = optim.Adam
                 ):
        """
        Args:
            observation_space: Agent's observations space.
            action_space: Agent's actions space.
            gamma: Value of gamma in the critic's target computation formulae.
            policy_update_frequency: how many critic update between policy updates.
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
        self.policy_update_frequency = policy_update_frequency
        self.learning_steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.batch_size = batch_size

        # Setup critic and its target
        self.critic = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, 1), nn.Tanh()
        ).to(self.device)
        critic_optimizer_class = optimizer if critic_optimizer is None else critic_optimizer
        critic_lr = learning_rate if critic_lr is None else critic_lr
        self.critic_optimizer = critic_optimizer_class(params=self.critic.parameters(), lr=critic_lr)
        self.critic_tau = tau if critic_tau is None else critic_tau
        self.target_critic = deepcopy(self.critic)

        # Setup actor and its target
        self.actor = nn.Sequential(
                nn.Linear(self.observation_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, self.action_size), nn.Tanh()
        ).to(self.device)
        actor_lr = learning_rate if actor_lr is None else actor_lr
        actor_optimizer_class = optimizer if actor_optimizer is None else actor_optimizer
        self.actor_optimizer = actor_optimizer_class(params=self.actor.parameters(), lr=actor_lr)
        self.actor_tau = tau if actor_tau is None else actor_tau
        self.target_actor = deepcopy(self.actor)

        self.action_noise = torch.distributions.normal.Normal(
            torch.zeros(self.action_size).to(self.device),
            torch.full((self.action_size,), self.exploration_noise_std).to(self.device)
        )

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

    def get_value(self, observations, actions=None):
        with torch.no_grad():
            if actions is None:
                actions = self.actor(observations)
            if isinstance(observations, np.ndarray):
                observations = torch.Tensor(observations)
            if isinstance(actions, np.ndarray):
                actions = torch.Tensor(actions)
            critic_value = self.critic(torch.concat((observations, actions), dim=-1))
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

            with torch.no_grad():
                target_actions = self.target_actor(new_states)
                target_actions = self.scale_action(target_actions, Box(-1, 1, (self.action_size,)))
                critic_value_ = self.target_critic(torch.concat((new_states, target_actions), dim=-1))
            critic_value = self.critic(torch.concat((states, actions), dim=-1))
            target = (rewards + self.gamma * (1 - dones) * critic_value_.squeeze()).view(self.batch_size, 1)
            critic_loss = torch.nn.functional.mse_loss(target, critic_value)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.learning_steps_done % self.policy_update_frequency == 0:
                self.learning_steps_done = 0
                actions = self.actor(states)
                actions = self.scale_action(actions, Box(-1, 1, (self.action_size,)))
                actor_loss = - self.critic(torch.concat((states, actions), dim=-1))
                actor_loss = torch.mean(actor_loss)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.target_critic = copy_weights(self.target_critic, self.critic, self.critic_tau)
                self.target_actor = copy_weights(self.target_actor, self.actor, self.actor_tau)
            self.learning_steps_done += 1
