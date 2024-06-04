# Goal conditioned deep Q-network
import copy
import numpy as np
import torch
from gym.spaces import Box
from torch.nn import ReLU, Tanh
from torch.nn.functional import normalize
from .value_based_agent import ValueBasedAgent
from torch import optim
from torch.nn import functional
from torch.distributions.normal import Normal
from typing import Union
from ..utils import NeuralNetwork


class SAC(ValueBasedAgent):

    name = "SAC"

    def __init__(self, 
                 observation_space, 
                 action_space,
                 actor_lr: float = 0.0005,
                 critic_lr: float = 0.0005,
                 alpha: Union[None, float] = None,
                 critic_alpha: float = 0.6,
                 actor_alpha: float = 0.05,
                 nb_gradient_steps: int = 1,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 layer1_size: int = 250,
                 layer2_size: int = 150,
                 reward_scale: int = 15,
                 ):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(observation_space, action_space)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        alpha = alpha
        self.critic_alpha = critic_alpha
        self.actor_alpha = actor_alpha
        self.nb_gradient_steps = nb_gradient_steps
        if alpha is not None:
            self.critic_alpha = alpha
            self.actor_alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.layer_1_size = layer1_size
        self.layer_2_size = layer2_size
        self.reward_scale = reward_scale

        self.policy_update_frequency = 2
        self.learning_step = 1

        self.min_std = -20
        self.max_std = 2

        self.actor = NeuralNetwork(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                         2 * self.nb_actions, Tanh(), learning_rate=self.actor_lr, optimizer_class=optim.Adam,
                         device=self.device).float()
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = MLP(self.observation_size + self.nb_actions, self.layer_1_size, ReLU(),
                          self.layer_2_size, ReLU(), 1, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                          device=self.device).float()
        self.target_critic = copy.deepcopy(self.critic)

        self.passed_logs = []

    def get_q_value(self, observation):
        with torch.no_grad():
            observation = torch.from_numpy(observation).to(self.device) if isinstance(observation, np.ndarray) else observation

            next_actions, _ = self.sample_action(observation, use_target_network=True)
            critic_input = torch.concat((observation, next_actions), dim=-1)
            q_values = self.target_critic.forward(critic_input).view(-1)
        return q_values

    def sample_action(self, actor_input, use_target_network=False, explore=True):
        actor_network = self.target_actor if use_target_network else self.actor

        if isinstance(actor_input, np.ndarray):
            actor_input = torch.from_numpy(actor_input).to(self.device)
        actor_input = normalize(actor_input, p=2., dim=-1)  # Tensor torch.float64

        # Forward
        actor_output = actor_network(actor_input)
        if len(actor_input.shape) > 1:  # It's a batch
            actions_means = actor_output[:, :self.nb_actions]
            actions_log_stds = actor_output[:, self.nb_actions:]
        else:
            actions_means = actor_output[:self.nb_actions]
            actions_log_stds = actor_output[self.nb_actions:]

        if self.under_test or not explore:
            return actions_means, None
        else:
            actions_log_stds = torch.clamp(actions_log_stds, min=self.min_std, max=self.max_std)
            actions_stds = torch.exp(actions_log_stds)
            actions_distribution = Normal(actions_means, actions_stds)

            raw_actions = actions_distribution.rsample() if explore or self.under_test else actions_means

            log_probs = actions_distribution.log_prob(raw_actions)
            actions = torch.tanh(raw_actions)
            log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(-1)
            actions = self.scale_action(actions, Box(-1, 1, (self.nb_actions,)))

            return actions, log_probs

    def action(self, observation, explore=True):
        with torch.no_grad():
            action, _ = self.sample_action(observation, explore=explore)
        return action.cpu().detach().numpy()

    def get_value(self, observations, actions=None):
        with torch.no_grad():
            if actions is None:
                actions, _ = self.sample_action(observations, explore=False)
            elif isinstance(actions, np.ndarray):
                actions = torch.tensor(actions)
            if isinstance(observations, np.ndarray):
                observations = torch.tensor(observations)
            critic_value = self.critic(torch.concat((observations, actions), dim=-1))
        if len(critic_value.shape) > 1:
            critic_value = critic_value.squeeze()
        return critic_value.detach().numpy()

    def learn(self):
        if not self.under_test and len(self.replay_buffer) > self.batch_size:
            for _ in range(self.nb_gradient_steps):
                observations, actions, rewards, new_observations, done = self.replay_buffer.sample(self.batch_size)

                # Training critic
                with torch.no_grad():
                    next_actions, next_log_probs = \
                        self.sample_action(new_observations, use_target_network=True)
                    critic_input = torch.concat((new_observations, next_actions), dim=-1)
                    self.passed_logs.append(next_log_probs)
                    next_q_values = \
                        self.target_critic(critic_input).view(-1)

                q_hat = self.reward_scale * rewards + self.gamma * (1 - done) * \
                    (next_q_values - self.critic_alpha * next_log_probs)
                q_values = self.critic(torch.concat((observations, actions), dim=-1)).view(-1)
                critic_loss = functional.mse_loss(q_values, q_hat)
                self.critic.learn(critic_loss)
                self.target_critic.converge_to(self.critic, tau=self.tau)

                if self.learning_step % self.policy_update_frequency == 0:
                    for _ in range(self.policy_update_frequency):
                        # Train actor
                        actions, log_probs = self.sample_action(observations)
                        log_probs = log_probs.view(-1)
                        critic_values = self.critic(torch.concat((observations, actions), dim=-1)).view(-1)

                        actor_loss = self.actor_alpha * log_probs - critic_values
                        actor_loss = torch.mean(actor_loss)
                        self.actor.learn(actor_loss, retain_graph=True)
                        self.target_actor.converge_to(self.actor, tau=self.tau)
                self.learning_step += 1
