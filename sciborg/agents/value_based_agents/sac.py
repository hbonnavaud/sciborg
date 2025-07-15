from copy import deepcopy
from typing import Union, Type

import numpy as np
import torch
from torch import optim, nn
from gymnasium.spaces import Box, Discrete

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


class SAC(ValueBasedAgent):
    OBSERVATION_SPACE_TYPE=Box

    def __init__(self, *args, **params):
        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's action space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
            gamma (float, optional): Discount factor used in the critic's target computation formula.
            # exploration_noise_std (float, optional): Standard deviation of the Gaussian noise added to the actor's
            #     output for exploration.
            policy_update_frequency (int, optional): Number of critic updates between each policy (actor) update.
            batch_size (int, optional): Number of samples in each training batch.
            replay_buffer_size (int, optional): Maximum number of interaction records the replay buffer can store
                before it is full.
            steps_before_learning (int, optional): Number of steps to take before learning begins, allowing the replay
                buffer to fill.
            target_network_update_frequency (int, optional): Interval of target update steps. Target networks updates
                are triggered every time the current step index modulo `target_network_update_frequency` equals zero.
            layer_1_size (int, optional): Size of the first layer in the actor, critic, and target networks.
            layer_2_size (int, optional): Size of the second layer in the actor, critic, and target networks.
            tau (float, optional): Soft update coefficient for the target actor and critic networks. If set, overrides
                both 'actor_tau' and 'critic_tau'.
            actor_tau (float, optional): Soft update coefficient for the target actor network. Overridden by 'tau' if
                it is set.
            critic_tau (float, optional): Soft update coefficient for the target critic network. Overridden by 'tau' if
                it is set.
            learning_rate (float, optional): Learning rate for both actor and critic networks. If set, overrides both
                'actor_lr' and 'critic_lr'.
            actor_lr (float, optional): Learning rate for the actor network. Overridden by 'learning_rate' if it is set.
            critic_lr (float, optional): Learning rate for the critic network. Overridden by 'learning_rate' if it is
                set.
            alpha_lr (float, optional): Learning rate for the alpha attribute. Overridden by 'learning_rate' if it is
                set.
            alpha (float, optional): Initial value for the 'alpha' attribute.
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for both actor and critic. If set,
                overrides both 'actor_optimizer_class' and 'critic_optimizer_class'.
            actor_optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for the actor network.
                Overridden by 'optimizer_class' if it is set.
            critic_optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for the critic network.
                Overridden by 'optimizer_class' if it is set.
            autotune_alpha (bool, optional): Boolean indicating whether the alpha attribute should be learned or not.
            target_entropy_scale (float, optional): Scale for the target entropy used for alpha finetuning. Unused if
                'autotune_alpha' is False.
        """

        super().__init__(*args, **params)
        self.name = params.get("name", "SAC")
        
        assert self.device is not None
        # Gather parameters
        self.gamma = params.get("gamma", 0.99)
        # self.exploration_noise_std = params.get("exploration_noise_std", 0.1)
        self.policy_update_frequency = params.get("policy_update_frequency", 2)
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 0)
        self.target_network_update_frequency = params.get("target_network_update_frequency", 1)
        self.layer_1_size = params.get("layer_1_size", 256)
        self.layer_2_size = params.get("layer_2_size", 256)
        self.tau = params.get("tau", None)
        self.actor_tau = params.get("actor_tau", 0.001)
        self.critic_tau = params.get("critic_tau", 0.001)
        self.learning_rate = params.get("learning_rate", None)
        self.actor_lr = params.get("actor_lr", 3e-4)
        self.critic_lr = params.get("critic_lr", 1e-3)
        self.alpha_lr = params.get("alpha_lr", 0.00025)
        self.alpha = params.get("alpha", 0.2)
        self.optimizer_class = params.get("optimizer_class", None)
        self.actor_optimizer_class = params.get("actor_optimizer_class", optim.Adam)
        self.critic_optimizer_class = params.get("critic_optimizer_class", optim.Adam)
        self.autotune_alpha = params.get("autotune_alpha", True)
        self.target_entropy_scale = params.get("target_entropy_scale", 0.89)

        if self.tau is not None:
            self.critic_tau = self.tau
            self.actor_tau = self.tau

        if self.learning_rate is not None:
            self.critic_lr = self.learning_rate
            self.actor_lr = self.learning_rate
            self.alpha_lr = self.learning_rate

        if self.optimizer_class is not None:
            self.critic_optimizer_class = self.optimizer_class
            self.actor_optimizer_class = self.optimizer_class

        # Instantiate the class
        self.learning_steps_done = 0
        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size, device=self.device)

        # Setup critic and its target
        self.critic_1 = nn.Sequential(
            nn.Linear(self.observation_size + self.action_size, self.layer_1_size), nn.ReLU(),
            nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
            nn.Linear(self.layer_2_size, 1)
        ).to(self.device)
        self.target_critic_1 = deepcopy(self.critic_1)

        self.critic_2 = nn.Sequential(
            nn.Linear(self.observation_size + self.action_size, self.layer_1_size), nn.ReLU(),
            nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
            nn.Linear(self.layer_2_size, 1)
        ).to(self.device)
        self.critic_optimizer = self.critic_optimizer_class(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.critic_lr, eps=1e-4)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Setup actor and its target
        self.actor = Actor(
            device=self.device,
            observation_size=self.observation_size,
            action_space=self.action_space,
            layer_1_size=self.layer_1_size,
            layer_2_size=self.layer_2_size,
        ).to(self.device)
        self.actor_optimizer = self.actor_optimizer_class(self.actor.parameters(), lr=self.actor_lr, eps=1e-4)

        # Information to scale the action to the action space
        self.action_scale = (torch.from_numpy((self.action_space.high - self.action_space.low) / 2.)
                             .to(device=self.device, dtype=torch.float32))
        self.action_offset = (torch.from_numpy((self.action_space.high + self.action_space.low) / 2.)
                              .to(device=self.device, dtype=torch.float32))

        # self.action_noise = torch.distributions.normal.Normal(
        #     torch.zeros(self.action_size).to(self.device),
        #     torch.full((self.action_size,), self.exploration_noise_std).to(self.device)
        # )

        if self.autotune_alpha:
            self.target_entropy = - self.target_entropy_scale * torch.log(1 / torch.tensor(self.action_size))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, eps=1e-4)

    def action(self, observation: np.ndarray, explore=True) -> np.ndarray:
        """
        Args:
            observation: The observation from which we want the agent to take an action.
            explore: Boolean indicating whether the agent can explore with this action of only exploit.
            If test_episode was set to True in the last self.start_episode call, the agent will exploit (explore=False)
            no matter the explore value here.
        Returns:
            np.ndarray: The action chosen by the agent.
        """
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor.get_action(observation)[0].to(self.device)
            # if not self.under_test and explore:  TODO: TEST without this shit Update the __init__ docstring for action_noise
            #     action += self.action_noise.sample()

            # Fit action to our action_space
            action = (action * self.action_scale + self.action_offset).cpu().detach().numpy()
        return action

    def process_interaction(self,
                            action: np.ndarray,
                            reward: float,
                            new_observation: np.ndarray,
                            done: bool,
                            learn: bool = True):
        """
        Processed the passed interaction using the given information.
        The state from which the action has been performed is kept in the agent's attribute, and updated everytime this function is called.
        Therefore, it does not appear in the function signature.
        Args:
            action (np.ndarray): the action performed by the agent at this step.
            reward (float): the reward returned by the environment following this action.
            new_observation (np.ndarray): the new state reached by the agent with this action.
            done (bool): whether the episode is done (no action will be performed from the given new_state) or not.
            learn (bool): whether the agent cal learn from this step or not (will define if the agent can save this interaction
                data, and start a learning step or not).
        """
        if learn and not self.under_test:
            self.replay_buffer.append((self.last_observation, action, reward, new_observation, done))
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def learn(self):
        """
        Trigger the agent learning process.
        Make sure that self.test_episode is False, otherwise, an error will be raised.
        """
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

            if self.learning_steps_done % self.target_network_update_frequency == 0:
                self.target_critic_1 = copy_weights(self.target_critic_1, self.critic_1, self.critic_tau)
                self.target_critic_2 = copy_weights(self.target_critic_2, self.critic_2, self.critic_tau)
            self.learning_steps_done += 1

    def get_value(self, observations: np.ndarray, actions: np.ndarray = None) -> np.ndarray:
        """
        Args:
            observations (np.ndarray): The observation(s) from which we want to obtain a value. Could be a batch.
            actions (np.ndarray, optional): The action that will be performed from the given observation(s). If none,
                the agent compute itself which action it would have taken from these observations.
        Returns:
            np.ndarray: The value of the given features.
        """
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if actions is None:
                actions, _ = self.actor(observations)
            if isinstance(observations, np.ndarray):
                observations = torch.Tensor(observations)
            if isinstance(actions, np.ndarray):
                actions = torch.Tensor(actions)
            critic_value = self.critic_1(torch.concat((observations, actions), dim=-1))
        return critic_value.flatten().cpu().detach().numpy()
