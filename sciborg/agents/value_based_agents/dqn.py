# Goal conditioned deep Q-network
import pickle
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
import copy
import numpy as np
import torch
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
from gymnasium.spaces import Discrete


class DQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    NAME = "DQN"
    OBSERVATION_SPACE_TYPE=Discrete

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
        # Gather parameters
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 0)
        self.learning_frequency = params.get("learning_frequency", 1)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.gamma = params.get("gamma", 0.95)
        self.layer_1_size = params.get("layer_1_size", 128)
        self.layer_2_size = params.get("layer_2_size", 84)
        self.initial_epsilon = params.get("initial_epsilon", 1)
        self.final_epsilon = params.get("final_epsilon", 0.05)
        self.steps_before_epsilon_decay = params.get("steps_before_epsilon_decay", 20)
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.model = params.get("model", None)
        self.optimizer_class = params.get("optimizer_class", torch.optim.Adam)
        self.criterion = params.get("criterion", torch.nn.functional.mse_loss)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.tau = params.get("tau", 0.0003)

        assert issubclass(self.optimizer_class, torch.optim.Optimizer)

        # Instantiate the class
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        self.epsilon = None
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_period
        self.total_steps = 0
        if self.model is None:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.observation_size, self.layer_1_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_1_size, self.layer_2_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_2_size, self.action_size)
            ).to(self.device)
        else:
            assert isinstance(self.model, torch.nn.Module)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.target_model = copy.deepcopy(self.model)

    def start_episode(self, observation, test_episode=False):
        self.epsilon = 1.0
        super().start_episode(observation, test_episode)

    def get_value(self, observations, actions=None):
        with torch.no_grad():
            values = self.model(observations)
            if actions is None:
                values = values.max(-1).values
            else:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions)
                values = values.gather(1, actions.to(torch.long).unsqueeze(1))
        return values.cpu().detach().numpy()

    def action(self, observation, explore=True):
        if explore and not self.under_test and self.train_interactions_done > self.steps_before_epsilon_decay:
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_step)

        if explore and not self.under_test and np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.action_space.n)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).to(self.device)
            with torch.no_grad():
                q_values = self.model(observation)
                action = torch.argmax(q_values).item()
        return action

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        if learn and not self.under_test:
            self.replay_buffer.append((self.last_observation, action, reward, new_observation, done))
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def learn(self):
        assert not self.under_test
        if (self.train_interactions_done >= self.steps_before_learning
                and self.train_interactions_done % self.learning_frequency == 0
                and len(self.replay_buffer) > self.batch_size):
            for _ in range(self.nb_gradient_steps):
                observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)
                q_prime = self.target_model(new_observations).max(1)[0].detach()
                q_target = rewards + self.gamma * (1 - dones) * q_prime
                q_values = self.model(observations).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_values, q_target.unsqueeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def set_device(self, device):
        super().set_device(device)
        self.model.to(device)
        self.target_model.to(device)
