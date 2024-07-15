from sciborg.agents.agent import Agent
from typing import Union
from gym.spaces import Box, Discrete
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

    def get_value(self, observation):
        return self.critic(self.features(observation)).squeeze()

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
    def __init__(self,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],

                 model: torch.nn.Module = None,
                 layer_1_size=256,
                 layer2_size=256,
                 layer3_size=256,

                 learning_rate: float = 2.5e-4,  # the learning rate of the optimizer"""
                 anneal_lr: bool = False,  # Toggle learning rate annealing for policy and value networks"""
                 gamma: float = 0.99,  # the discount factor gamma"""
                 gae_lambda: float = 0.95,  # the lambda for the general advantage estimation"""
                 update_epochs: int = 4,  # the K epochs to update the policy"""
                 use_normalisation_advantage: bool = True,  # Toggles advantages normalization"""
                 clip_coefficient: float = 0.1,  # the surrogate clipping coefficient"""
                 clip_value_function_loss: bool = True,  # Toggles whether to use a clipped loss for the value function, as per the paper."""
                 entropy_coefficient: float = 0.01,  # coefficient of the entropy"""
                 value_function_coefficient: float = 0.5,  # coefficient of the value function"""
                 max_grad_norm: float = 0.5,  # the maximum norm for the gradient clipping"""
                 target_kl: float = None,  # the target KL divergence threshold"""

                 # to be filled in runtime
                 min_buffer_size: int = 1000,  # the batch size (computed in runtime)"""
                 num_mini_batches: int = 4,  # the number of mini-batches"""

                 ):
        super().__init__(observation_space, action_space)

        self.model = model
        if self.model is None:
            self.model = Model(self.observation_size, self.nb_actions, self.device,
                               layer_1_size, layer2_size, layer3_size)
        else:
            mandatory_functions = ["get_action", "get_value", "get_action_and_value"]
            for function_name in mandatory_functions:
                assert hasattr(model, function_name) and callable(getattr(model, function_name)), (
                    "The model given to discrete ppo have no function called {}, which is mandatory."
                    .format(function_name))

        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.nb_mini_batches = num_mini_batches
        self.update_epochs = update_epochs
        self.use_normalisation_advantage = use_normalisation_advantage
        self.clip_coefficient = clip_coefficient
        self.clip_value_function_loss = clip_value_function_loss
        self.entropy_coefficient = entropy_coefficient
        self.value_function_coefficient = value_function_coefficient
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.buffer_size = min_buffer_size

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

    def action(self, observation, explore=True):
        with torch.no_grad():
            return self.model.get_action(observation, explore=explore).detach().cpu().item()

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        if learn:
            self.store_transition(self.last_observation, action, reward, new_observation, done)
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
