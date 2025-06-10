import copy
from typing import Union
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer


class DQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    name = "DQN"

    def __init__(self, *args, **params):
        super().__init__(*args, **params)
        
        # Gather parameters
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 10000)
        self.learning_frequency = params.get("learning_frequency", 10)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.gamma = params.get("gamma", 0.95)
        self.layer_1_size = params.get("layer_1_size", 128)
        self.layer_2_size = params.get("layer_2_size", 84)
        self.initial_epsilon = params.get("initial_epsilon", 1)
        self.final_epsilon = params.get("final_epsilon", 0.05)
        self.steps_before_epsilon_decay = params.get("steps_before_epsilon_decay", 20)
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.model = params.get("model", None)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.criterion = params.get("criterion", functional.mse_loss)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.tau = params.get("tau", 2.5e-4)

        # Instantiate the class
        self.replay_buffer = ReplayBuffer(replay_buffer_size, self.device)
        self.epsilon = None
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_period
        self.total_steps = 0
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(self.observation_size, layer_1_size),
                nn.ReLU(),
                nn.Linear(layer_1_size, layer_2_size),
                nn.ReLU(),
                nn.Linear(layer_2_size, self.action_space.n)
            )
        else:
            assert isinstance(model, torch.nn.Module)
            self.model = model
        assert issubclass(optimizer, optim.Optimizer)
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.target_model = copy.deepcopy(self.model)

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
            with torch.no_grad():
                q_values = self.model(observation)
                action = torch.argmax(q_values).item()
        return action

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
                self.model.learn(loss)

            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
