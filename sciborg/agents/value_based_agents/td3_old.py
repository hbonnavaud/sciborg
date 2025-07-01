from copy import deepcopy
from typing import Union, Type
import numpy as np
import torch
from torch import optim, nn
from gymnasium.spaces import Box

from ..utils import ReplayBuffer, NeuralNetwork
from .value_based_agent import ValueBasedAgent


class TD3(ValueBasedAgent):
    NAME = "TD3"
    OBSERVATION_SPACE_TYPE=Box

    def __init__(self, *args, **params):
        """
        Args:
            observation_space: Agent's observations space.
            action_space: Agent's actions space.
            gamma: Value of gamma in the critic's target computation formulae.
            layer_1_size: Size of actor and critic first hidden layer.
            layer_2_size: Size of actor and critic second hidden layer.
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
        super().__init__(*args, **params)

        # Gather parameters
        self.gamma = params.get("gamma", 0.99)
        self.exploration_noise_std = params.get("exploration_noise_std", 0.1)
        self.target_action_noise_std = params.get("target_action_noise_std", 0.05)
        self.target_action_max_noise = params.get("target_action_max_noise", 0.1)
        self.policy_update_frequency = params.get("policy_update_frequency", 2)
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 0)
        self.layer_1_size = params.get("layer_1_size", 256)
        self.layer_2_size = params.get("layer_2_size", 256)
        self.tau = params.get("tau", None)
        self.actor_tau = params.get("actor_tau", 0.001)
        self.critic_tau = params.get("critic_tau", 0.001)
        self.learning_rate = params.get("learning_rate", None)
        self.actor_lr = params.get("actor_lr", 2.5e-5)
        self.critic_lr = params.get("critic_lr", 1e-4)
        self.optimizer_class = params.get("optimizer_class", None)
        self.actor_optimizer_class = params.get("actor_optimizer_class", optim.Adam)
        self.critic_optimizer_class = params.get("critic_optimizer_class", optim.Adam)
        
        if self.optimizer_class:
            assert issubclass(self.optimizer_class, optim.Optimizer), "The optimizer should be a subclass of torch.optim.Optimizer"
            self.critic_optimizer_class = self.optimizer_class
            self.actor_optimizer_class = self.optimizer_class
        else:
            assert issubclass(self.critic_optimizer_class, optim.Optimizer), "The critic optimizer should be a subclass of torch.optim.Optimizer"
            assert issubclass(self.actor_optimizer_class, optim.Optimizer), "The actor optimizer should be a subclass of torch.optim.Optimizer"

        if self.learning_rate:
            assert isinstance(self.learning_rate, float)
            self.critic_lr = self.learning_rate
            self.actor_lr = self.learning_rate
        else:
            assert isinstance(self.critic_lr, float)
            assert isinstance(self.actor_lr, float)

        if self.tau:
            assert isinstance(self.tau, float)
            self.critic_tau = self.tau
            self.actor_tau = self.tau
        else:
            assert isinstance(self.critic_tau, float)
            assert isinstance(self.actor_tau, float)

        # Instantiate the class
        self.learning_steps_done = 0
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)

        # Setup critic and its target
        self.critic_1 = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, self.layer_1_size), nn.ReLU(),
                nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
                nn.Linear(self.layer_2_size, 1)).to(self.device)
        self.target_critic_1 = deepcopy(self.critic_1)

        self.critic_2 = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, self.layer_1_size), nn.ReLU(),
                nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
                nn.Linear(self.layer_2_size, 1)).to(self.device)
        self.target_critic_2 = deepcopy(self.critic_2)

        self.critic_optimizer = self.critic_optimizer_class(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.critic_lr
        )

        self.critic_1_losses = []
        self.critic_2_losses = []
        self.actor_losses = []

        # Setup actor and its target
        # self.actor = NeuralNetwork(
        #     module=nn.Sequential(
        #         nn.Linear(self.observation_size, self.layer_1_size), nn.ReLU(),
        #         nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
        #         nn.Linear(self.layer_2_size, self.action_size), nn.Tanh()),
        #     device=self.device,
        #     tau=self.actor_tau,
        #     learning_rate=self.actor_lr,
        #     optimizer_class=self.actor_optimizer_class
        # )

        self.actor = nn.Sequential(
                nn.Linear(self.observation_size, self.layer_1_size), nn.ReLU(),
                nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
                nn.Linear(self.layer_2_size, self.action_size), nn.Tanh()
        ).to(self.device)
        self.actor_optimizer = self.actor_optimizer_class(params=self.actor.parameters(), lr=self.actor_lr)
        self.target_actor = deepcopy(self.actor)

        self.action_noise = torch.distributions.normal.Normal(
            torch.zeros(self.action_size).to(self.device), 
            torch.full((self.action_size,), self.exploration_noise_std).to(self.device)
        )

    def action(self, observation, explore=True):

        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor(observation)
            if not self.under_test and explore:
                action += self.action_noise.sample()
                action = torch.clamp(action, -1.0, 1.0)
            action = action.cpu().detach().numpy()

            # Fit action to our action_space
            # action = self.scale_action(action, Box(-1, 1, (self.action_size,)))
        return action

    def store_interaction(self, *interaction_data):
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
            rewards = torch.clamp(rewards, -10, 10)

            with torch.no_grad():
                
                # clipped_noise = torch.randn_like(actions, device=self.device) * self.target_action_noise_std
                # clipped_noise = torch.clamp(clipped_noise, -self.target_action_max_noise, self.target_action_max_noise)
                # target_actions = self.target_actor(new_states) + clipped_noise
                #
                # target_actions = torch.clamp(target_actions,
                #                             torch.tensor(self.action_space.low).to(self.device),
                #                             torch.tensor(self.action_space.high).to(self.device))
                #
                # target_actions = self.scale_action(target_actions, Box(-1, 1, (self.action_size,)))

                noise = torch.randn_like(actions).to(self.device) * self.target_action_noise_std
                noise = torch.clamp(noise, -self.target_action_max_noise, self.target_action_max_noise)
                target_actions = self.target_actor(new_states) + noise
                target_actions = torch.clamp(target_actions, -1.0, 1.0)

                target_critic_value_1 = self.target_critic_1(torch.concat((new_states, target_actions), dim=-1))
                target_critic_value_2 = self.target_critic_2(torch.concat((new_states, target_actions), dim=-1))
                target_critic_value = torch.min(target_critic_value_1, target_critic_value_2)

            target = (rewards + self.gamma * (1 - dones) * target_critic_value.squeeze()).view(self.batch_size, 1)
            critic_1_value = self.critic_1(torch.concat((states, actions), dim=-1))
            critic_2_value = self.critic_2(torch.concat((states, actions), dim=-1))

            critic_1_loss = torch.nn.functional.mse_loss(critic_1_value, target)
            critic_2_loss = torch.nn.functional.mse_loss(critic_2_value, target)
            critic_loss = critic_1_loss + critic_2_loss
            self.critic_1_losses.append(critic_1_loss.item())
            self.critic_2_losses.append(critic_2_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)

            if self.learning_steps_done % self.policy_update_frequency == 0:
                actions = self.actor(states)
                # actions = self.scale_action(actions, Box(-1, 1, (self.action_size,)))
                actor_loss = - self.critic_1(torch.cat((states, actions), dim=-1)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_losses.append(actor_loss.item())

                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(self.actor_tau * param.data + (1 - self.actor_tau) * target_param.data)

            self.learning_steps_done += 1