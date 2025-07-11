from copy import deepcopy
import numpy as np
import torch
from torch import optim, nn
from gymnasium.spaces import Box
from ..utils import ReplayBuffer
from .value_based_agent import ValueBasedAgent

from torch.optim import Adam
class DDPG(ValueBasedAgent):
    OBSERVATION_SPACE_TYPE=Box

    def __init__(self, *args, **params):
        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): The environment's action space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
            gamma (float, optional): Discount factor used in the critic's target computation formula.
            exploration_noise_std (float, optional): Standard deviation of the Gaussian noise added to the actor's
                output for exploration.
            policy_update_frequency (int, optional): Number of critic updates between each policy (actor) update.
            batch_size (int, optional): Number of samples in each training batch.
            replay_buffer_size (int, optional): Maximum number of interaction records the replay buffer can store
                before it is full.
            steps_before_learning (int, optional): Number of steps to take before learning begins, allowing the replay
                buffer to fill.
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
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for both actor and critic. If set,
                overrides both 'actor_optimizer_class' and 'critic_optimizer_class'.
            actor_optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for the actor network.
                Overridden by 'optimizer_class' if it is set.
            critic_optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer class for the critic network.
                Overridden by 'optimizer_class' if it is set.
        """
        super().__init__(*args, **params)
        self.name = params.get("name", "DDPG")

        # Gather parameters
        self.gamma = params.get("gamma", 0.99)
        self.exploration_noise_std = params.get("exploration_noise_std", 0.2)
        self.policy_update_frequency = params.get("policy_update_frequency", 2)
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 100)
        self.layer_1_size = params.get("layer_1_size", 256)
        self.layer_2_size = params.get("layer_2_size", 256)
        self.tau = params.get("tau", None)
        self.actor_tau = params.get("actor_tau", 0.001)
        self.critic_tau = params.get("critic_tau", 0.001)
        self.learning_rate = params.get("learning_rate", None)
        self.actor_lr = params.get("actor_lr", 3e-4)
        self.critic_lr = params.get("critic_lr", 3e-4)
        self.optimizer_class = params.get("optimizer_class", None)
        self.actor_optimizer_class = params.get("actor_optimizer_class", optim.Adam)
        self.critic_optimizer_class = params.get("critic_optimizer_class", optim.Adam)

        if self.tau is not None:
            self.critic_tau = self.tau
            self.actor_tau = self.tau

        if self.learning_rate is not None:
            self.critic_lr = self.learning_rate
            self.actor_lr = self.learning_rate
            
        if self.optimizer_class is not None:
            self.critic_optimizer_class = self.optimizer_class
            self.actor_optimizer_class = self.optimizer_class

        # Instantiate the class
        self.learning_steps_done = 0
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        # Setup critic and its target
        self.critic = nn.Sequential(
                nn.Linear(self.observation_size + self.action_size, self.layer_1_size), nn.ReLU(),
                nn.Linear(self.layer_1_size, self.layer_2_size), nn.ReLU(),
                nn.Linear(self.layer_2_size, 1)
        ).to(self.device)
        self.critic_optimizer = self.critic_optimizer_class(params=self.critic.parameters(), lr=self.critic_lr)
        self.target_critic = deepcopy(self.critic)

        # Setup actor and its target
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

        self.action_scale = (torch.from_numpy((self.action_space.high - self.action_space.low) / 2.)
                             .to(device=self.device, dtype=torch.float32))
        self.action_offset = (torch.from_numpy((self.action_space.high + self.action_space.low) / 2.)
                              .to(device=self.device, dtype=torch.float32))

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
        if self.train_interactions_done <= self.steps_before_learning:
            return self.action_space.sample()

        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).to(self.device)
            action = self.actor(observation)
            if not self.under_test and explore:
                action += self.action_noise.sample()
                action = torch.clamp(action,
                                     torch.from_numpy(self.action_space.low).to(self.device),
                                     torch.from_numpy(self.action_space.high).to(self.device))
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
            states, actions, rewards, new_states, dones = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target_actions = self.target_actor(new_states) * self.action_scale + self.action_offset
                critic_value_ = self.target_critic(torch.concat((new_states, target_actions), dim=-1))
            critic_value = self.critic(torch.concat((states, actions), dim=-1))
            target = (rewards + self.gamma * (1 - dones) * critic_value_.squeeze()).view(self.batch_size, 1)
            critic_loss = torch.nn.functional.mse_loss(critic_value, target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.learning_steps_done % self.policy_update_frequency == 0:
                self.learning_steps_done = 0
                actions = self.actor(states) * self.action_scale + self.action_offset
                actor_loss = - self.critic(torch.concat((states, actions), dim=-1)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.critic_tau * param.data + (1 - self.critic_tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    target_param.data.copy_(self.actor_tau * param.data + (1 - self.actor_tau) * target_param.data)
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
        with torch.no_grad():
            if actions is None:
                actions = self.actor(observations) * self.action_scale + self.action_offset
            if isinstance(observations, np.ndarray):
                observations = torch.Tensor(observations)
            if isinstance(actions, np.ndarray):
                actions = torch.Tensor(actions)
            critic_value = self.critic(torch.concat((observations, actions), dim=-1))
        return critic_value.flatten().detach().numpy()
