# Goal conditioned deep Q-network
import pickle
from agents.value_based_agents.value_based_agent import ValueBasedAgent
from ..utils.nn import MLP
import copy
import numpy as np
import torch
from torch import optim
from torch.nn import ReLU


class DQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    name = "DQN"

    def __init__(self, observation_space, action_space, **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(observation_space, action_space, **params)

        self.gamma = params.get("gamma", 0.95)
        self.layer_1_size = params.get("layer_1_size", 64)
        self.layer_2_size = params.get("layer_2_size", 64)
        self.epsilon_min = params.get("epsilon_min", 0.01)
        self.epsilon_max = params.get("epsilon_max", 1.)
        self.epsilon_decay_delay = params.get("epsilon_decay_delay", 20)
        self.epsilon = None
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.model = params.get("model", None)

        #  NEW, goals will be stored inside the replay buffer. We need a specific one with enough place to do so
        self.learning_rate = params.get("learning_rate", 0.001)
        self.steps_before_target_update = params.get("steps_before_target_update", 1)
        self.steps_since_last_target_update = 0
        self.tau = params.get("tau", 0.001)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)

        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.total_steps = 0

        # NEW, The input observation size is multiplied by two because we need to also take the goal as input
        if self.model is None:
            self.model = MLP(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                             self.nb_actions, learning_rate=self.learning_rate, optimizer_class=optim.Adam,
                             device=self.device).float()

        self.criterion = torch.nn.SmoothL1Loss()
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.epsilon = self.epsilon_max

    def set_device(self, device):
        self.device = device
        self.model.to(device)
        self.target_model.to(device)

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
        if self.training_steps_done > self.epsilon_decay_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if explore and not self.under_test and np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.nb_actions)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            with torch.no_grad():
                q_values = self.model(observation)
                action = torch.argmax(q_values).item()
        return action

    def learn(self):
        assert not self.under_test
        for _ in range(self.nb_gradient_steps):
            if len(self.replay_buffer) > self.batch_size:
                observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)
                q_prime = self.target_model(new_observations).max(1)[0].detach()
                update = rewards + self.gamma * (1 - dones) * q_prime
                q_s_a = self.model(observations).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_s_a, update.unsqueeze(1))
                self.model.learn(loss)

        self.steps_since_last_target_update += 1
        if self.steps_since_last_target_update >= self.steps_before_target_update:

            self.target_model.converge_to(self.model, self.tau)
            self.steps_since_last_target_update = 0

    def save(self, directory):
        super().save(directory)

        with open(directory + "observation_space.pkl", "wb") as f:
            pickle.dump(self.observation_space, f)
        with open(directory + "action_space.pkl", "wb") as f:
            pickle.dump(self.action_space, f)
        with open(directory + "init_params.pkl", "wb") as f:
            pickle.dump(self.init_params, f)

        torch.save(self.model, directory + "model.pt")
        torch.save(self.target_model, directory + "target_model.pt")

        with open(directory + "replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load(self, directory):
        super().load(directory)

        with open(directory + "observation_space.pkl", "rb") as f:
            self.observation_space = pickle.load(f)
        with open(directory + "action_space.pkl", "rb") as f:
            self.action_space = pickle.load(f)
        with open(directory + "init_params.pkl", "rb") as f:
            self.init_params = pickle.load(f)
        self.reset()

        self.model = torch.load(directory + "model.pt")
        self.target_model = torch.load(directory + "target_model.pt")

        with open(directory + "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)
