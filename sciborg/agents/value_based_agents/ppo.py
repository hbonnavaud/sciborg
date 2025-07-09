from sciborg.agents.rlagent import Agent
from typing import Union
from gymnasium.spaces import Box, Discrete
import torch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PpoReplayBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

        self.values = []
        self.log_probs = []
        self.entropy = []
        self.next_values = []
        self.advantages = []
        self.returns = []

        self._size = 0

    def __getattribute__(self, item):
        if item.startswith('_'):
            raise AttributeError("Item {} cannot be accessed.".format(item))
        else:
            super().__getattribute__(item)

    def append(self, observation, action, log_probs, reward, next_observation, done, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)
        self.values.append(value)
        self.size += 1

    def get_mini_batches(self, nb_mini_batches):
        min_mini_batches_size = self.size // 4
        mini_batches = []
        start = 0
        for mini_batch_id in range(nb_mini_batches):
            end = start + min_mini_batches_size
            mini_batches.append([
                self.observations[start:end],
                self.actions[start:end],
                self.log_probs[start:end],
                self.rewards[start:end],
                self.next_observations[start:end],
                self.dones[start:end],
                self.values[start:end],
                self.advantages[start:end],
                self.returns[start:end]
            ])

        # We have divided the replay buffer into n equal parts, but this division leaves a remainder that we need to
        # distribute among the mini-batches already formed.
        for i in range(self.size - start):
            mini_batches[i][0].append(self.observations[start + i])
            mini_batches[i][1].append(self.actions[start + i])
            mini_batches[i][2].append(self.log_probs[start + i])
            mini_batches[i][3].append(self.rewards[start + i])
            mini_batches[i][4].append(self.next_observations[start + i])
            mini_batches[i][5].append(self.dones[start + i])
            mini_batches[i][6].append(self.values[start + i])
            mini_batches[i][7].append(self.advantages[start + i])
            mini_batches[i][8].append(self.returns[start + i])

        # Now we have our mini batches ready to be returned, we can reset our replay buffer.
        self.__init__()

        return mini_batches


class Model(torch.nn.Module):
    def __init__(self, observation_size, nb_actions, device, layer_1_size, layer2_size, layer3_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.model = torch.nn.Sequential(
            layer_init(torch.nn.Linear(observation_size, layer_1_size), std=0.01), torch.nn.ReLU(),
            layer_init(torch.nn.Linear(layer_1_size, layer2_size), std=0.01), torch.nn.ReLU(),
            layer_init(torch.nn.Linear(layer2_size, layer3_size), std=0.01), torch.nn.ReLU(),
        ).to(self.device)
        self.actor = layer_init(torch.nn.Linear(layer3_size, nb_actions), std=0.01).to(self.device)
        self.critic = layer_init(torch.nn.Linear(layer3_size, 1), std=1).to(self.device)

    def features(self, observation):
        if isinstance(observation, (np.ndarray)):
            observation = torch.tensor(observation)
        observation = observation.to(self.device, dtype=torch.float32)
        return self.model(observation)

    def get_value(self, observations: np.ndarray, actions: np.ndarray = None):
        """
        Args:
            observations: the observation(s) from which we want to obtain a value. Could be a batch.
            observations: the observation(s) from which we want to obtain a value. Could be a batch.
            actions: the action that will be performed from the given observation(s). If none, the agent compute itself
                which action it would have taken from these observations.
        Returns: the value of the given features.
        """
        return self.critic(self.features(observations)).squeeze()

    def get_action(self, observation, explore=True):
        logits = self.actor(self.features(observation))
        if explore:
            return torch.distributions.Categorical(logits=logits).sample()
        else:
            return torch.argmax(logits).item()

    def get_action_and_value(self, observation, action=None):
        hidden = self.features(observation)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).squeeze()


class PpoDiscreteAgent(Agent):
    OBSERVATION_SPACE_TYPE=Discrete
    
    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        self.name = params.get("name", "PPO")

        # Gather parameters
        self.model = params.get("model", None)
        self.layer_1_size = params.get("layer_1_size", 256)
        self.layer2_size = params.get("layer2_size", 256)
        self.layer3_size = params.get("layer3_size", 256)
        self.learning_rate = params.get("learning_rate", 2.5e-4)
        self.anneal_lr = params.get("anneal_lr", False)
        self.gamma = params.get("gamma", 0.99)
        self.gae_lambda = params.get("gae_lambda", 0.95)
        self.update_epochs = params.get("update_epochs", 4)
        self.use_normalisation_advantage = params.get("use_normalisation_advantage", True)
        self.clip_coefficient = params.get("clip_coefficient", 0.1)
        self.clip_value_function_loss = params.get("clip_value_function_loss", True)
        self.entropy_coefficient = params.get("entropy_coefficient", 0.01)
        self.value_function_coefficient = params.get("value_function_coefficient", 0.5)
        self.max_grad_norm = params.get("max_grad_norm", 0.5)
        self.target_kl = params.get("target_kl", None)
        self.min_buffer_size = params.get("min_buffer_size", 1000)
        self.num_mini_batches = params.get("num_mini_batches", 4)

        # Instantiate the class
        if self.model is None:
            self.model = Model(self.observation_size, self.nb_actions, self.device,
                               self.layer_1_size, self.layer2_size, self.layer3_size)
        else:
            mandatory_functions = ["get_action", "get_value", "get_action_and_value"]
            for function_name in mandatory_functions:
                assert hasattr(model, function_name) and callable(getattr(model, function_name)), (
                    "The model given to discrete ppo have no function called {}, which is mandatory."
                    .format(function_name))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_observations = []
        self.nb_learning_data_available = 0

    def get_value(self, observations):
        return self.model.get_value(observations)

    def set_device(self, device):
        self.model.to(device=device)

    def action(self, observation: np.ndarray, explore=True):
        """
        Args:
            observation: The observation from which we want the agent to take an action.
            explore: Boolean indicating whether the agent can explore with this action of only exploit.
            If test_episode was set to True in the last self.start_episode call, the agent will exploit (explore=False)
            no matter the explore value here.
        Returns: The action chosen by the agent.
        """
        with torch.no_grad():
            return self.model.get_action(observation, explore=explore).detach().cpu().item()

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

    # def stop_episode(self):
    #     self.learn()
    #     super().stop_episode()

    def store_transition(self, observation, action, reward, next_observation, done):
        if self.nb_learning_data_available >= self.buffer_size:
            self.observations[self.nb_learning_data_available % self.buffer_size] = observation
            self.actions[self.nb_learning_data_available % self.buffer_size] = action
            self.rewards[self.nb_learning_data_available % self.buffer_size] = reward
            self.dones[self.nb_learning_data_available % self.buffer_size] = done
            self.next_observations[self.nb_learning_data_available % self.buffer_size] = next_observation
        else:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_observations.append(next_observation)
        self.nb_learning_data_available += 1

    def save_interaction(self, observation, action, reward, next_observation, done):
        self.store_transition(observation, action, reward, next_observation, done)

    def learn(self):
        """
        Trigger the agent learning process.
        Make sure that self.test_episode is False, otherwise, an error will be raised.
        """
        if self.nb_learning_data_available >= self.buffer_size:

            observations = torch.tensor(np.array(self.observations)).float().to(self.device)
            next_observations = torch.tensor(np.array(self.next_observations)).float().to(self.device)
            actions = torch.tensor(np.array(self.actions)).float().to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                _, log_probs, entropy, values = self.model.get_action_and_value(observations)
                next_values = self.model.get_value(next_observations).squeeze()
                advantages = torch.zeros(self.nb_learning_data_available).to(self.device)
                last_gae_lambda = 0
                for t in reversed(range(self.nb_learning_data_available)):
                    next_non_terminal = 1.0 - self.dones[t]
                    delta = self.rewards[t] + self.gamma * next_values[t] * next_non_terminal - values[t]
                    last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                    advantages[t] = last_gae_lambda
                returns = advantages + values

            indexes = np.arange(len(self.rewards))
            minimal_batch_size, rest = divmod(len(self.rewards), self.nb_mini_batches)
            # `-> The rest will be distributed into the first batches.
            for epoch in range(self.update_epochs):
                np.random.shuffle(indexes)
                for mini_batch_id in range(self.nb_mini_batches):
                    def buffer_index(batch_index):
                        return minimal_batch_size * batch_index + min(batch_index, rest)
                        # `-> The + min(...) stuff is to include the rest
                    start = buffer_index(mini_batch_id)
                    end = buffer_index(mini_batch_id + 1)

                    # In the following variables, mb_ stands for mini_batch
                    mb_indexes = indexes[start:end]
                    mb_observations = observations[mb_indexes]
                    mb_actions = actions[mb_indexes]
                    mb_log_probs = log_probs[mb_indexes]
                    mb_values = values[mb_indexes]
                    mb_returns = returns[mb_indexes]
                    mb_advantages = advantages[mb_indexes]

                    _, new_log_prob, entropy, new_value = (
                        self.model.get_action_and_value(mb_observations, mb_actions.long()))
                    log_ratio = new_log_prob - mb_log_probs
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = (ratio - 1 - log_ratio).mean()

                    mb_advantages = mb_advantages
                    if self.use_normalisation_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coefficient, 1 + self.clip_coefficient)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    new_value = new_value.view(-1)
                    if self.clip_value_function_loss:
                        v_loss_unclipped = (new_value - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            new_value - mb_values,
                            -self.clip_coefficient,
                            self.clip_coefficient,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.entropy_coefficient * entropy_loss + v_loss * self.value_function_coefficient

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            # Clear learning data
            self.observations = []
            self.actions = []
            self.rewards = []
            self.dones = []
            self.next_observations = []
            self.nb_learning_data_available = 0
