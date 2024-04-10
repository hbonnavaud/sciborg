import pickle
from copy import deepcopy

import numpy as np
import torch
from gym.spaces import Box
from torch.nn import ReLU, Tanh

from utils import create_dir
from agents.value_based_agents.value_based_agent import ValueBasedAgent
from ..utils.nn import MLP
from torch import optim


class DDPG(ValueBasedAgent):
    name = "DDPG"

    def __init__(self, state_space, action_space, **params):
        """
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(state_space, action_space, **params)

        self.actor_lr = params.get("actor_lr", 0.000025)
        self.critic_lr = params.get("critic_lr", 0.00025)
        self.tau = params.get("tau", 0.001)
        self.gamma = params.get("gamma", 0.99)
        self.layer_1_size = params.get("layer1_size", 200)
        self.layer_2_size = params.get("layer2_size", 150)
        self.noise_std = params.get("noise_std", 0.1)
        self.steps_before_target_update = params.get("steps_before_target_update", 5)
        self.steps_since_last_update = 0

        self.actor = MLP(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                         self.nb_actions, Tanh(), learning_rate=self.actor_lr, optimizer_class=optim.Adam,
                         device=self.device).float()
        self.critic = MLP(self.observation_size + self.nb_actions, self.layer_1_size, ReLU(),
                          self.layer_2_size, ReLU(), 1, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                          device=self.device).float()

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.normal_distribution = torch.distributions.normal.Normal(
            torch.zeros(self.nb_actions), torch.full((self.nb_actions,), self.noise_std))

    def set_device(self, device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

    def action(self, observation, explore=True):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor(observation).to(self.device)
            if not self.under_test and explore:
                action += self.normal_distribution.sample()
            action = action.cpu().detach().numpy()

            # Fit action to our action_space
            action = self.scale_action(action, Box(-1, 1, (self.nb_actions,)))
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
            critic_value = self.critic(torch.concat((observations, actions), dim=-1))
        return critic_value.flatten().detach().numpy()

    def learn(self):
        assert not self.under_test
        if not self.under_test and len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, new_states, dones = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target_actions = self.target_actor(new_states)
                target_actions = self.scale_action(target_actions, Box(-1, 1, (self.nb_actions,)))
                critic_value_ = self.target_critic(torch.concat((new_states, target_actions), dim=-1))
            critic_value = self.critic(torch.concat((states, actions), dim=-1))
            # target = torch.addcmul(rewards, self.gamma, 1 - dones, critic_value_.squeeze()).view(self.batch_size, 1)
            target = (rewards + self.gamma * (1 - dones) * critic_value_.squeeze()).view(self.batch_size, 1)
            critic_loss = torch.nn.functional.mse_loss(target, critic_value)
            self.critic.learn(critic_loss)

            actions = self.actor(states)
            actions = self.scale_action(actions, Box(-1, 1, (self.nb_actions,)))
            actor_loss = - self.critic(torch.concat((states, actions), dim=-1))
            actor_loss = torch.mean(actor_loss)
            self.actor.learn(actor_loss)

            self.steps_since_last_update += 1
            if self.steps_since_last_update % self.steps_before_target_update == 0:
                self.target_critic.converge_to(self.critic, tau=self.tau)
                self.target_actor.converge_to(self.actor, tau=self.tau)

    def save(self, directory):
        super().save(directory)

        torch.save(self.critic, directory + "critic.pt")
        torch.save(self.target_critic, directory + "target_critic.pt")

        torch.save(self.actor, directory + "actor.pt")
        torch.save(self.target_actor, directory + "target_actor.pt")

        with open(directory + "replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load(self, directory):
        super().load(directory)

        self.critic = torch.load(directory + "critic.pt")
        self.target_critic = torch.load(directory + "target_critic.pt")

        self.actor = torch.load(directory + "actor.pt")
        self.target_actor = torch.load(directory + "target_actor.pt")

        with open(directory + "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)
