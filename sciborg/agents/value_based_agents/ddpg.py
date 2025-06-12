import warnings
from copy import deepcopy
from typing import Union, Type
import numpy as np
import torch
from torch import optim, nn
from gymnasium.spaces import Box
from ..utils import ReplayBuffer
from .value_based_agent import ValueBasedAgent
from ..utils.copy_weights import copy_weights
import inspect


class DDPG(ValueBasedAgent):
    NAME = "DDPG"
    OBSERVATION_SPACE_TYPE=Box

    def __init__(self, *args, **params):
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
        super().__init__(*args, **params)

        # Gather parameters
        self.gamma = params.get("gamma", 0.99)
        self.exploration_noise_std = params.get("exploration_noise_std", 0.1)
        self.policy_update_frequency = params.get("policy_update_frequency", 2)
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 1000)
        self.layer1_size = params.get("layer1_size", 256)
        self.layer2_size = params.get("layer2_size", 256)
        self.tau = params.get("tau", None)
        self.critic_tau = params.get("critic_tau", 0.001)
        self.actor_tau = params.get("actor_tau", 0.001)
        self.learning_rate = params.get("learning_rate", None)
        self.critic_lr = params.get("critic_lr", 0.00025)
        self.actor_lr = params.get("actor_lr", 0.000025)
        self.optimizer = params.get("optimizer", None)
        self.critic_optimizer = params.get("critic_optimizer", optim.Adam)
        self.actor_optimizer = params.get("actor_optimizer", optim.Adam)

        if self.tau is not None:
            self.critic_tau = self.tau
            self.actor_tau = self.tau

        if self.learning_rate is not None:
            self.critic_lr = self.learning_rate
            self.actor_lr = self.learning_rate
            
        if self.optimiser is not None:
            self.critic_optimizer = self.optimiser
            self.actor_optimizer = self.optimiser

        # Instantiate the class
        self.learning_steps_done = 0
        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        # Setup critic and its target
        self.critic = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, 1), nn.Tanh()
        ).to(self.device)
        self.critic_optimizer = self.critic_optimizer(params=self.critic.parameters(), lr=self.critic_lr)
        self.target_critic = deepcopy(self.critic)

        # Setup actor and its target
        self.actor = nn.Sequential(
                nn.Linear(self.observation_size, layer1_size), nn.ReLU(),
                nn.Linear(layer1_size, layer2_size), nn.ReLU(),
                nn.Linear(layer2_size, self.action_size), nn.Tanh()
        ).to(self.device)
        self.actor_optimizer = self.actor_optimizer(params=self.actor.parameters(), lr=self.actor_lr)
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
