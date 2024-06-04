from copy import deepcopy
from typing import Union, Type

import numpy as np
import torch
from torch import optim, nn
from gym.spaces import Box, Discrete

from ..utils import ReplayBuffer, NeuralNetwork
from .value_based_agent import ValueBasedAgent


class Actor(NeuralNetwork):
    def __init__(self, observation_size: int, action_space: Union[Box, Discrete], layer_1_size: int, layer_2_size: int,
                 tau: float, learning_rate: float, optimizer_class: Type[optim.Optimizer]):
        super().__init__(tau=tau, learning_rate=learning_rate, optimizer_class=optimizer_class)
        self.fc1 = nn.Linear(observation_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc_mean = nn.Linear(layer_2_size, np.prod(action_space.shape))
        self.fc_log_std = nn.Linear(layer_2_size, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.format_input(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
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
        log_prob = log_prob.sum(1, keepdim=True)
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

                 alpha: float = 0.2,
                 autotune_alpha: bool = True,
                 target_entropy_scale: float = 0.89,
                 alpha_lr: Union[None, float] = None
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
        self.learning_steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.batch_size = batch_size

        # Setup critic and its target
        self.critic_1 = NeuralNetwork(
            tau=critic_tau if tau is None else tau,
            learning_rate=critic_lr if learning_rate is None else learning_rate,
            optimizer_class=critic_optimizer if optimizer is None else optimizer,
            module=nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size),
                nn.ReLU(),
                nn.Linear(layer1_size, layer2_size),
                nn.ReLU(),
                nn.Linear(layer2_size, 1),
                nn.Tanh()
            )
        )
        self.target_critic_1 = deepcopy(self.critic_1)

        self.critic_2 = NeuralNetwork(
            tau=critic_tau if tau is None else tau,
            learning_rate=critic_lr if learning_rate is None else learning_rate,
            optimizer_class=critic_optimizer if optimizer is None else optimizer,
            module=nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, layer1_size),
                nn.ReLU(),
                nn.Linear(layer1_size, layer2_size),
                nn.ReLU(),
                nn.Linear(layer2_size, 1),
                nn.Tanh()
            )
        ).to(self.device)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Setup actor and its target
        self.actor = Actor(
            observation_size=self.observation_size,
            action_space=self.action_space,
            layer_1_size=self.layer_1_size,
            layer_2_size=self.layer_2_size,
            tau=actor_tau if tau is None else tau,
            learning_rate=actor_lr if learning_rate is None else learning_rate,
            optimizer_class=actor_optimizer if optimizer is None else optimizer
        ).to(self.device)

        self.action_noise = torch.distributions.normal.Normal(
            torch.zeros(self.action_size), torch.full((self.action_size,), self.exploration_noise_std))

        self.autotune_alpha = autotune_alpha
        self.alpha = alpha
        if self.autotune_alpha:
            self.target_entropy = - target_entropy_scale * torch.log(1 / torch.tensor(self.action_size))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = log_alpha.exp().item()
            self.a_optimizer = optim.Adam([log_alpha], lr=alpha_lr, eps=1e-4)

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

            ######################## CLEAN RL LEARNING
            data = rb.sample(args.batch_size)
            # CRITIC training
            with torch.no_grad():
                _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations)
                qf2_next_target = qf2_target(data.next_observations)
                # we can use the action probabilities instead of MC sampling to estimate the expectation
                min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                # adapt Q-target for discrete Q-function
                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

            # use Q-values only for the taken actions
            qf1_values = qf1(data.observations)
            qf2_values = qf2(data.observations)
            qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
            qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # ACTOR training
            _, log_pi, action_probs = actor.get_action(data.observations)
            with torch.no_grad():
                qf1_values = self.critic_1(data.observations)
                qf2_values = self.critic_2(data.observations)
                min_qf_values = torch.min(qf1_values, qf2_values)
            # no need for reparameterization, the expectation can be calculated for discrete actions
            actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

            self.actor.learn(actor_loss)

            ######################## ALPHA AUTOTUNE
            if self.autotune_alpha:
                # re-use action probabilities for temperature loss
                alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
        self.learning_steps_done += 1