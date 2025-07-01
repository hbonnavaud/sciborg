# Goal conditioned deep Q-network
import pickle
from .value_based_agent import ValueBasedAgent
from ..utils import ReplayBuffer
from ...utils.nn import MLP
import copy
import numpy as np
import torch
from torch import optim
from torch.nn import ReLU


def soft_max(q_values, tau):
    return torch.nn.functional.softmax(q_values / tau, dim=-1)


class MunchausenDQN(ValueBasedAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """

    name = "Munchausen_DQN"

    def __init__(self, observation_space, action_space, **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(observation_space, action_space, **params)


        self.min_samples_before_learn = params.get("min_samples_before_learn", 100)
        self.batch_size = params.get("batch_size", 256)
        self.buffer_max_size = params.get("buffer_max_size", int(1e6))

        self.replay_buffer = ReplayBuffer(self.buffer_max_size, self.device)

        self.gamma = params.get("gamma", 0.98)
        self.layer_1_size = params.get("layer_1_size", 128)
        self.layer_2_size = params.get("layer_2_size", 128)
        self.model = params.get("model", None)
        self.tau_soft = params.get("tau_soft", 0.03)

        #  NEW, goals will be stored inside the replay buffer. We need a specific one with enough place to do so
        self.learning_rate = params.get("learning_rate", 5e-4)
        self.steps_before_learn = params.get("steps_before_learn", 100)
        self.steps_before_target_update = params.get("steps_before_target_update", 1)
        self.steps_since_last_target_update = 0
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)

        self.tau = params.get("tau", 0.001)

        self.epsilon_tar = params.get("epsilon_tar", 1e-6)
        """the epsilon term for numerical stability"""
        self.alpha = params.get("alpha", 0.9)
        """the entropy regularization parameter"""
        self.l_0 = params.get("l_0", -1.0)
        """the lower bound of the weighted log probability"""

        self.total_steps = 0

        # NEW, The input observation size is multiplied by two because we need to also take the goal as input
        if self.model is None:
            self.model = MLP(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                             self.action_size, learning_rate=self.learning_rate, optimizer_class=optim.Adam,
                             device=self.device).float()

        self.criterion = torch.nn.SmoothL1Loss()
        self.target_model = copy.deepcopy(self.model).to(self.device)

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
        assert observation.shape == self.observation_space.shape
        # greedy_action(self.model, observation) function in RL5 notebook
        with torch.no_grad():
            q_values = self.model(observation)
            policy = soft_max(q_values, self.tau_soft)
            action = torch.multinomial(policy, 1).squeeze(-1).cpu().numpy()
        return action

    def learn(self):
        assert not self.under_test

        for _ in range(self.nb_gradient_steps):
            if len(self.replay_buffer) > self.batch_size:
                observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
                actions = actions.to(dtype=torch.int64)[:, np.newaxis]

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
                self.model.learn(loss)

        self.steps_since_last_target_update += 1
        if self.steps_since_last_target_update >= self.steps_before_target_update:

            self.target_model.converge_to(self.model, self.tau)
            self.steps_since_last_target_update = 0

    def process_interaction(self, action, reward, new_observation, done, learn=True, old_observation=None):
        if done:
            debug = 1
        if learn and not self.under_test:
            old_observation = self.last_observation if old_observation is None else old_observation
            self.save_interaction(old_observation, action, reward, new_observation, done)
            self.learn()
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def save_interaction(self, *interaction_data):
        """
        Function that is called to ask our agent to learn about the given interaction. This function is separated from
        self.on_action_stop(**interaction_data) because we can imagine agents that do not learn on every interaction, or
        agents that learn on interaction they didn't make (like HER that add interaction related to fake goals in their
        last trajectory).
        on_action_stop is used to update variables likes self.last_observation or self.simulation_time_step_id, and
        learn_interaction is used to know the set of interactions we can learn about.

        Example: Our implementation of HER show a call to 'learn_interaction' without 'on_action_stop'
        (two last lines of 'her' file).
        """
        assert not self.under_test
        for data in interaction_data:
            if isinstance(data, np.ndarray):
                assert not np.isnan(data).any()
            else:
                assert data is not None
        self.replay_buffer.append(interaction_data)

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
