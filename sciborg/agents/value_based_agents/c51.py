# Goal conditioned deep Q-network
import copy
from typing import Union
import numpy as np
import torch
from torch import optim, nn
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, 
                 observation_size, 
                 nb_actions, 
                 layer_1_size: int = 120, 
                 layer_2_size: int = 84,
                 nb_atoms: int = 101, 
                 q_value_min: float = -100, 
                 q_value_max: float = 100,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super().__init__()
        self.nb_atoms = nb_atoms
        self.register_buffer("atoms", torch.linspace(q_value_min, q_value_max, steps=nb_atoms))
        self.nb_actions = nb_actions
        self.network = nn.Sequential(
            nn.Linear(observation_size, layer_1_size),
            nn.ReLU(),
            nn.Linear(layer_1_size, layer_2_size),
            nn.ReLU(),
            nn.Linear(layer_2_size, self.nb_actions * nb_atoms),
        )

    def get_action(self, x, action=None):
        model_output = self.network(x)
        probability_mass_function = torch.softmax(model_output.view(len(x), self.nb_actions, self.n_atoms), dim=2)
        q_values = (probability_mass_function * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, probability_mass_function[torch.arange(len(x)), action]


class C51(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    name = "C51"

    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Gather parameters
        self.batch_size = params.get("batch_size", 256)
        self.replay_buffer_size = params.get("replay_buffer_size", int(1e6))
        self.steps_before_learning = params.get("steps_before_learning", 10000)
        self.learning_frequency = params.get("learning_frequency", 10)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.gamma = params.get("gamma", 0.95)
        self.layer_1_size = params.get("layer_1_size", 120)
        self.layer_2_size = params.get("layer_2_size", 84)
        self.initial_epsilon = params.get("initial_epsilon", 1)
        self.final_epsilon = params.get("final_epsilon", 0.05)
        self.steps_before_epsilon_decay = params.get("steps_before_epsilon_decay", 20)
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.nb_atoms = params.get("nb_atoms", 101)
        self.q_value_min = params.get("q_value_min", -100)
        self.q_value_max = params.get("q_value_max", 100)
        self.model = params.get("model", None)
        self.optimizer = params.get("optimizer", optim.Adam)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.tau = params.get("tau", 2.5e-4)

        # Instantiate the class
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        self.epsilon = None
        self.epsilon_step = (self.initial_epsilon - self.final_epsilon) / self.epsilon_decay_period
        self.total_steps = 0

        if model is None:
            self.model = QNetwork(self.observation_size, self.action_space.n,
                                  layer_1_size=self.layer_1_size, layer_2_size=self.layer_2_size,
                                  nb_atoms=self.nb_atoms, q_value_min=self.q_value_min, q_value_max=self.q_value_max)
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
                action, _ = self.model.get_action(observation).cpu().numpy()
        return action

    def learn(self):
        assert not self.under_test
        if (self.train_interactions_done >= self.steps_before_learning
                and self.train_interactions_done % self.learning_frequency == 0
                and len(self.replay_buffer) > self.batch_size):

            for _ in range(self.nb_gradient_steps):
                observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)

                # The code bellow is highly inspired from clean_rl
                with torch.no_grad():
                    _, next_proba_mass_functions = self.target_model.get_action(new_observations)
                    next_atoms = rewards + self.gamma * self.target_model.atoms * (1 - dones)

                    # projection
                    delta_z = self.target_model.atoms[1] - self.target_model.atoms[0]
                    tz = next_atoms.clamp(self.q_value_min, self.q_value_max)
                    b = (tz - self.q_value_min) / delta_z
                    l = b.floor().clamp(0, self.nb_atoms - 1)
                    u = b.ceil().clamp(0, self.nb_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_proba_mass_functions
                    d_m_u = (b - l) * next_proba_mass_functions
                    target_pmfs = torch.zeros_like(next_proba_mass_functions)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_proba_mass_functions = self.model.get_action(observations, actions.flatten())
                loss = (-(target_pmfs * old_proba_mass_functions.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                # optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
