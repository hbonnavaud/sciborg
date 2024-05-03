import pickle
from copy import deepcopy
import numpy as np
import torch
from torch.nn import ReLU, Tanh
from .dqn import DQN
from ..utils.nn import MLP
from torch import optim
import torch.nn.functional as F


class DistributionalDQN(DQN):
    """
    A distributional RL version of DQN, but implemented in the same way that SORB authors describe a
    distributional version of DDPG in SORB's paper (Eysenbach, 2017).
    """

    name = "Dist. DQN"

    def __init__(self, observation_space, action_space, **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(observation_space, action_space, **params)
        self.nb_models = params.get("nb_models", 3)
        self.out_dist_size = params.get("output_distribution_size", 40)
        self.out_dist_abscissa = - np.linspace(0, self.out_dist_size, self.out_dist_size)
        self.out_dist_abscissa_steps = self.out_dist_size / (self.out_dist_size - 1)
        self.models = params.get("models", None)

        if self.models is None:
            self.models = [
                MLP(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                             self.nb_actions * self.out_dist_size, learning_rate=self.learning_rate, optimizer_class=optim.Adam,
                             device=self.device).float() for _ in range(self.nb_models)
            ]
        self.target_models = [deepcopy(model) for model in self.models]
        self.errors = {}
        for i in range(self.nb_models):
            self.errors["model_" + str(i)] = []

        # Remove the attributes of the mother class that we will not use
        del self.model
        del self.target_model

    def set_device(self, device):
        self.device = device
        for m in self.models:
            m.to(device)
        for m in self.target_models:
            m.to(device)

    def get_q_values_probabilities(self, observations, use_target=False):
        is_batch = len(observations.shape) == len(self.observation_shape) + 1
        batch_size = observations.shape[0] if is_batch else 1
        models = self.target_models if use_target else self.models
        if is_batch:
            new_shape = (batch_size, self.nb_actions, self.out_dist_size)
        else:
            new_shape = (self.nb_actions, self.out_dist_size)
        q_values_probs = []
        for model_id in range(self.nb_models):
            value = models[model_id](observations)
            value = value.reshape(new_shape)
            value = F.softmax(value, dim=-1)
            q_values_probs.append(value)
        return torch.stack(q_values_probs).mean(0)

    def get_value(self, observations, actions=None, use_target=False):
        with torch.no_grad():
            q_values_probs = self.get_q_values_probabilities(observations, use_target)
            q_values_per_action = (q_values_probs * torch.tensor(self.out_dist_abscissa).to(self.device)).sum(dim=-1)
            if actions is None:
                values = q_values_per_action.max(-1).values
            else:
                if not isinstance(actions, torch.Tensor):
                    actions = torch.tensor(actions)
                values = q_values_per_action.gather(1, actions.to(torch.long).unsqueeze(1))
            return values.cpu().detach().numpy()

    def action(self, observation, explore=True):
        if not self.under_test:
            if self.training_steps_done > self.epsilon_decay_delay:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if explore and not self.under_test and np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.nb_actions)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            with torch.no_grad():
                q_values_probs = self.get_q_values_probabilities(observation)
                q_values_per_action = (q_values_probs * torch.tensor(self.out_dist_abscissa).to(self.device)).sum(dim=-1)
                action = q_values_per_action.max(-1).indices.detach().cpu().item()
        return action

    def learn(self):
        assert not self.under_test
        if len(self.replay_buffer) > self.batch_size:
            observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target_q_values_probs = self.get_q_values_probabilities(new_observations, use_target=True)

                target_q_values_per_action = (target_q_values_probs * torch.tensor(self.out_dist_abscissa).to(self.device)).sum(dim=-1)
                next_actions = target_q_values_per_action.max(-1).indices
                next_actions = next_actions.to(torch.long).repeat((self.out_dist_size, 1)).T.unsqueeze(1)
                target_q_values_probabilities = target_q_values_probs.gather(1, next_actions).squeeze()

            # Compute target

            # Compute distribution target
            # - Target distributions for samples where the goal has been reached
            #   shape: [[1, 0, 0, ...],
            #           [1, 0, 0, ...], ... ]

            reached_target_distribution = F.one_hot(torch.zeros(self.batch_size).to(dtype=torch.long),
                                                    self.out_dist_size).to(self.device)

            # - Target distribution for samples where the goal hasn't been reached
            #    * Build a column of 0
            first_column = torch.zeros(self.batch_size).unsqueeze(1).to(self.device)
            #    * Build others columns by shifting the target q_values
            middle_columns = target_q_values_probabilities[:, :-2]
            #    * Build the last column as the sum of the last two left q_values
            last_column = target_q_values_probabilities[:, -2:].sum(-1).unsqueeze(1)
            failed_target_distribution = torch.concat((first_column, middle_columns, last_column), -1)

            # Compute the target distributions
            target_distribution = torch.where(dones[:, np.newaxis] == 1.,
                                              reached_target_distribution,
                                              failed_target_distribution)

            for model_id in range(self.nb_models):
                model = self.models[model_id]
                model_distribution = model(observations).reshape((self.batch_size, self.nb_actions, self.out_dist_size))
                a = actions.to(torch.long).repeat((self.out_dist_size, 1)).T.unsqueeze(1)
                model_distribution = model_distribution.gather(1, a).squeeze()
                model_loss = F.cross_entropy(model_distribution, target_distribution)
                model.learn(model_loss)
                loss_copy = model_loss.detach().cpu().item()
                self.errors["model_" + str(model_id)].append(loss_copy)
                self.target_models[model_id].converge_to(model, tau=self.tau)

    def save(self, directory):
        super().save(directory)

        for model_id, model in enumerate(self.models):
            torch.save(model, directory + "model_" + str(model_id) + ".pt")

        with open(directory + "replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def save(self, directory):
        super().save(directory)

        for model_id, model in enumerate(self.models):
            self.models[model_id] = torch.load(directory + "model_" + str(model_id) + ".pt")

        with open(directory + "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)
