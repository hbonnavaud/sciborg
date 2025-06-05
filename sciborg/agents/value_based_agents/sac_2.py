from copy import deepcopy
from typing import Union, Type

import numpy as np
import torch
from torch import optim, nn
from gym.spaces import Box, Discrete

from ..utils import ReplayBuffer
from .value_based_agent import ValueBasedAgent
from ..utils.copy_weights import copy_weights


class Actor(torch.nn.Module):
    def __init__(self, observation_size: int, action_space: Union[Box, Discrete], layer_1_size: int, layer_2_size: int,
                 device=torch.device("cuda")):
        super().__init__()
        self.dtype = torch.float32
        self.device = device
        self.fc1 = nn.Linear(observation_size, layer_1_size).to(self.dtype).to(self.device)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size).to(self.dtype).to(self.device)
        self.fc_mean = nn.Linear(layer_2_size, np.prod(action_space.shape)).to(self.dtype).to(self.device)
        self.fc_log_std = nn.Linear(layer_2_size, np.prod(action_space.shape)).to(self.dtype).to(self.device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device).to(self.dtype)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_log_std(x))
        log_std_min = -5
        log_std_max = 2
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC_2(ValueBasedAgent):
    name = "SAC_2"

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
                 steps_before_learning: int = 1000,
                 target_network_frequency: int = 1,

                 layer1_size: int = 256,
                 layer2_size: int = 256,

                 tau: float = 0.005,

                 learning_rate: Union[None, float] = None,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 alpha_lr: float = 0.00025,

                 alpha: float = 0.2,
                 autotune_alpha: bool = True,
                 target_entropy_scale: float = 0.89,
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
            learning_rate: Learning rate of actor and critic modules. If set, this value overwrite both
                'actor_lr' and 'critic_lr' hyperparameters.
            actor_lr: Actor learning rate. Overwritten by 'learning_rate' if it is set.
            critic_lr: Critic learning rate. Overwritten by 'learning_rate' if it is set.
            alpha: (float) entropy regularisation hyperparameter.
            autotune_alpha: (bool) Whether to autotune alpha hyperparameter or not.
            target_entropy_scale: (float) Scale of the alpha autotune.
            alpha_lr: (float) Alpha autotune learning rate.
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
        self.target_network_frequency = target_network_frequency
        self.learning_steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.batch_size = batch_size

        # Setup critic and its target
        self.tau = tau
        critic_lr = critic_lr if learning_rate is None else learning_rate
        self.critic_1 = nn.Sequential(
            nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
            nn.Linear(layer1_size, layer2_size), nn.ReLU(),
            nn.Linear(layer2_size, 1), nn.Tanh()
        ).to(self.device)

        self.critic_2 = nn.Sequential(
            nn.Linear(self.observation_size + self.action_size, layer1_size), nn.ReLU(),
            nn.Linear(layer1_size, layer2_size), nn.ReLU(),
            nn.Linear(layer2_size, 1)
        ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                                 lr=critic_lr, 
                                                 eps=1e-4)

        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Setup actor and its target
        self.actor = Actor(
            device=self.device,
            observation_size=self.observation_size,
            action_space=self.action_space,
            layer_1_size=self.layer_1_size,
            layer_2_size=self.layer_2_size,
        ).to(self.device)
        actor_lr = actor_lr if learning_rate is None else learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-4)

        self.action_noise = torch.distributions.normal.Normal(
            torch.zeros(self.action_size).to(self.device),
            torch.full((self.action_size,), self.exploration_noise_std).to(self.device)
        )

        self.autotune_alpha = autotune_alpha
        self.alpha = alpha
        if self.autotune_alpha:
            self.target_entropy = - target_entropy_scale * torch.log(1 / torch.tensor(self.action_size))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            alpha_lr = alpha_lr if learning_rate is None else learning_rate
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, eps=1e-4)

    def action(self, observation, explore=True):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor.get_action(observation)[0].to(self.device)
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
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)

            # CRITIC training
            with torch.no_grad():
                next_actions, next_log_pi, _ = self.actor.get_action(next_observations)
                critic_1_next_value = self.target_critic_1(torch.cat((next_observations, next_actions), -1))
                critic_2_next_value = self.target_critic_2(torch.cat((next_observations, next_actions), -1))
                min_qf_next_target = torch.min(critic_1_next_value, critic_2_next_value) - self.alpha * next_log_pi
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

            # use Q-values only for the taken actions
            critic_1_value = self.critic_1(torch.cat((observations, actions), -1)).view(-1)
            critic_2_value = self.critic_2(torch.cat((observations, actions), -1)).view(-1)
            critic_1_loss = torch.nn.functional.mse_loss(critic_1_value, next_q_value)
            critic_2_loss = torch.nn.functional.mse_loss(critic_2_value, next_q_value)
            qf_loss = critic_1_loss + critic_2_loss
            self.critic_optimizer.zero_grad()
            qf_loss.backward()
            self.critic_optimizer.step()

            if self.learning_steps_done % self.policy_update_frequency == 0:
                for _ in range(self.policy_update_frequency):
                    # ACTOR TRAINING
                    actions, log_actions, _ = self.actor.get_action(observations)
                    critic_1_value = self.critic_1(torch.cat((observations, actions), -1))
                    critic_2_value = self.critic_2(torch.cat((observations, actions), -1))
                    min_critic_value = torch.min(critic_1_value, critic_2_value)
                    actor_loss = ((self.alpha * log_actions) - min_critic_value).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # ALPHA AUTOTUNE
                    if self.autotune_alpha:
                        # re-use action probabilities for temperature loss
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(observations)
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                        alpha_loss = alpha_loss.mean()

                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            if self.learning_steps_done % self.target_network_frequency == 0:
                self.target_critic_1 = copy_weights(self.target_critic_1, self.critic_1, self.tau)
                self.target_critic_2 = copy_weights(self.target_critic_2, self.critic_2, self.tau)
            self.learning_steps_done += 1
