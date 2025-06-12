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


def soft_max(q_values, tau):
    return torch.nn.functional.softmax(q_values / tau, dim=-1)


class MunchausenDQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    NAME = "Munchausen_DQN"
    OBSERVATION_SPACE_TYPE=Discrete

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
        self.optimizer_class = params.get("optimizer_class", torch.optim.Adam)
        self.criterion = params.get("criterion", torch.nn.functional.mse_loss)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.tau = params.get("tau", 0.0003)
        self.tau_soft = params.get("tau_soft", 0.03)
        self.steps_before_learn = params.get("steps_before_learn", 0)
        self.target_network_update_frequency = params.get("target_network_update_frequency", 1)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)
        self.epsilon_tar = params.get("epsilon_tar", 1e-6)
        self.alpha = params.get("alpha", 0.9)
        self.l_0 = params.get("l_0", -1.0)

        assert issubclass(self.optimizer_class, torch.optim.Optimizer)

        # Instantiate the class
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.device)
        self.total_steps = 0
        if self.model is None:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.observation_size, self.layer_1_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_1_size, self.layer_2_size), torch.nn.ReLU(),
                torch.nn.Linear(self.layer_2_size, self.action_size), torch.nn.Tanh()
            ).to(self.device)
        else:
            assert isinstance(self.model, torch.nn.Module)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.target_model = copy.deepcopy(self.model)
        self.nb_learning_steps_without_target_update = 0

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
        assert observation.shape[-1] == self.observation_space.n
        if isinstance(observation, np.ndarray):
            observation = torch.Tensor(observation).to(device=self.device)
        # greedy_action(self.model, observation) function in RL5 notebook
        with torch.no_grad():
            q_values = self.model(observation)
            policy = soft_max(q_values, self.tau_soft)
            action = torch.multinomial(policy, 1).squeeze(-1).cpu().numpy()
        return action

    def process_interaction(self, action, reward, new_observation, done, learn=True):
        if learn and not self.under_test:
            self.replay_buffer.append((self.last_observation, action, reward, new_observation, done))
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def learn(self):
        assert not self.under_test

        for _ in range(self.nb_gradient_steps):
            if len(self.replay_buffer) > self.batch_size:
                observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
                actions = actions.to(dtype=torch.int64)[:, np.newaxis]

                # # OLD
                # q_prime = self.target_model(next_observations).max(1)[0].detach()
                # update = rewards + self.gamma * (1 - dones) * q_prime
                # q_s_a = self.model(observations).gather(1, actions.to(torch.long).unsqueeze(1))
                # loss = self.criterion(q_s_a, update.unsqueeze(1))
                # self.model.learn(loss)

                # NEW
                with torch.no_grad():
                    target_q_values = self.target_model(observations)
                    target_policy = soft_max(target_q_values, self.tau_soft)
                    target_next_q_values = self.target_model(next_observations)
                    target_next_policy = soft_max(target_next_q_values, self.tau_soft)
                    log_part = torch.log(target_policy.gather(1, actions).squeeze())
                    red_term = self.alpha * (self.tau_soft * log_part + self.epsilon_tar).clamp(self.l_0, 0.0)
                    bleu_term = -self.tau_soft * torch.log(target_next_policy + self.epsilon_tar)
                    munchausen_target = rewards + red_term + self.gamma * (1 - dones) * \
                        (target_next_policy * (target_next_q_values + bleu_term)).sum(dim=-1)
                    td_target = munchausen_target.squeeze()

                old_val = self.model(observations).gather(1, actions).squeeze()
                loss = torch.nn.functional.mse_loss(td_target, old_val)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.nb_learning_steps_without_target_update + 1 >= self.target_network_update_frequency:
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.nb_learning_steps_without_target_update = 0
        else:
            self.nb_learning_steps_without_target_update += 1

    def save(self, directory):
        super().save(directory)

        with open(str(directory) + "observation_space.pkl", "wb") as f:
            pickle.dump(self.observation_space, f)
        with open(str(directory) + "action_space.pkl", "wb") as f:
            pickle.dump(self.action_space, f)
        with open(str(directory) + "init_params.pkl", "wb") as f:
            pickle.dump(self.init_params, f)

        torch.save(self.model, str(directory) + "model.pt")
        torch.save(self.target_model, str(directory) + "target_model.pt")

        with open(str(directory) + "replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load(self, directory):
        super().load(directory)

        with open(str(directory) + "observation_space.pkl", "rb") as f:
            self.observation_space = pickle.load(f)
        with open(str(directory) + "action_space.pkl", "rb") as f:
            self.action_space = pickle.load(f)
        with open(str(directory) + "init_params.pkl", "rb") as f:
            self.init_params = pickle.load(f)
        self.reset()

        self.model = torch.load(str(directory) + "model.pt")
        self.target_model = torch.load(str(directory) + "target_model.pt")

        with open(str(directory) + "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)

    def set_device(self, device):
        super().set_device(device)
        self.model.to(device)
        self.target_model.to(device)
