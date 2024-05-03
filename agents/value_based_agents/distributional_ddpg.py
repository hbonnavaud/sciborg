import pickle
from copy import deepcopy
import numpy as np
import torch
from torch.nn import ReLU, Tanh
from torch import optim
import torch.nn.functional as F
from .ddpg import DDPG
from ..utils.nn import MLP


class DistributionalDDPG(DDPG):
    """
    A distributional RL version of DDPG as defined in SORB (Eysenbach, 2017).
    """

    name = "Dist. DDPG"

    def __init__(self, observation_space, action_space, **params):
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(observation_space, action_space, **params)

        # Set up parameters as described in sorb paper
        self.batch_size = params.get("batch_size", 64)
        self.layer_1_size = params.get("layer1_size", 64)
        self.layer_2_size = params.get("layer2_size", 64)
        self.replay_buffer_max_size = params.get("replay_buffer_max_size", 1e6)
        self.actor_lr = params.get("actor_lr", 1e-4)
        self.critic_lr = params.get("critic_lr", 1e-4)
        self.tau = params.get("tau", 0.05)
        self.noise_std = params.get("noise_std", 0.1)
        self.steps_before_target_update = params.get("steps_before_target_update", 5)
        self.nb_critics = params.get("nb_critics", 1)
        self.out_dist_size = params.get("output_distribution_size", 20)
        self.out_dist_abscissa = - np.linspace(0, self.out_dist_size, self.out_dist_size)
        self.out_dist_abscissa_steps = self.out_dist_size / (self.out_dist_size - 1)

        self.actor = MLP(self.observation_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                         self.nb_actions, Tanh(), learning_rate=self.actor_lr, optimizer_class=optim.Adam,
                         device=self.device).float()
        self.target_actor = deepcopy(self.actor)

        self.critics = [
            MLP(self.observation_size + self.nb_actions, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                self.out_dist_size, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                device=self.device).float() for _ in range(self.nb_critics)
        ]
        self.target_critics = [deepcopy(self.critics[i]) for i in range(self.nb_critics)]

    def set_device(self, device):
        self.device = device
        self.actor.to(device)
        self.target_actor.to(device)
        for c in self.critics:
            c.to(device)
        for c in self.target_critics:
            c.to(device)

    def get_value(self, observations, actions=None, use_target=False, gradient=False):
        with torch.set_grad_enabled(gradient):
            if actions is None:
                actions = self.actor(observations).detach().numpy()
            critics = self.target_critics if use_target else self.critics
            critic_inputs = np.concatenate((observations, actions), -1)
            q_values_probs = [F.softmax(critic(critic_inputs), dim=-1) for critic in critics]
            q_values_probs = torch.stack(q_values_probs).min(0).values
            q_values = (q_values_probs * torch.tensor(self.out_dist_abscissa)).sum(dim=-1)
        return q_values if gradient else q_values.detach().cpu().numpy()

    def learn(self):
        assert not self.under_test
        if len(self.replay_buffer) > self.batch_size:
            observations, actions, rewards, new_observations, dones = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                target_actions = self.target_actor.forward(new_observations)
                target_critic_inputs = np.concatenate((new_observations, target_actions), -1)
                target_q_values_probabilities = [F.softmax(critic(target_critic_inputs), dim=-1)
                                                 for critic in self.target_critics]
                target_q_values_probabilities = torch.stack(target_q_values_probabilities).mean(0)

            # Compute target

            # Compute distribution target
            # - Target distributions for samples where the goal has been reached
            #   shape: [[1, 0, 0, ...],
            #           [1, 0, 0, ...], ... ]
            reached_target_distribution = F.one_hot(torch.zeros(self.batch_size).to(dtype=torch.long),
                                                    self.out_dist_size)

            # - Target distribution for samples where the goal hasn't been reached
            #    * Build a column of 0
            first_column = torch.zeros(self.batch_size).unsqueeze(1)
            #    * Build others columns by shifting the target q_values
            middle_columns = target_q_values_probabilities[:, :-2]
            #    * Build the last column as the sum of the last two left q_values
            last_column = target_q_values_probabilities[:, -2:].sum(-1).unsqueeze(1)
            failed_target_distribution = torch.concat((first_column, middle_columns, last_column), -1)

            # Compute the target distributions
            target_distribution = torch.where(dones[:, np.newaxis] == 1.,
                                              reached_target_distribution,
                                              failed_target_distribution)

            for critic_id in range(self.nb_critics):
                critic = self.critics[critic_id]
                critic_distribution = critic(torch.concat((observations, actions), dim=-1))
                critic_loss = F.cross_entropy(critic_distribution, target_distribution)
                critic.learn(critic_loss)
                self.target_critics[critic_id].converge_to(critic, tau=self.tau)

            # Train actor
            actions = self.actor(observations)
            actor_input = torch.concat((observations, actions), dim=-1)
            q_values_probs = [F.softmax(critic(actor_input), dim=-1) for critic in self.critics]
            q_values_probs = torch.stack(q_values_probs).min(0).values
            q_values = (q_values_probs * torch.tensor(self.out_dist_abscissa)).sum(dim=-1)
            self.actor.learn(torch.mean(- q_values))
            self.target_actor.converge_to(self.actor, tau=self.tau)

    def save(self, directory):
        super().save(directory)

        for critic_id, critic in enumerate(self.critics):
            torch.save(critic, directory + "critic_" + str(critic_id) + ".pt")

        for target_critic_id, target_critic in enumerate(self.target_critics):
            torch.save(target_critic, directory + "target_critic_" + str(target_critic_id) + ".pt")

        torch.save(self.actor, directory + "actor.pt")
        torch.save(self.target_actor, directory + "target_actor.pt")

        with open(directory + "replay_buffer.pkl", "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load(self, directory):
        super().load(directory)

        for critic_id, critic in enumerate(self.critics):
            self.critics[critic_id] = torch.load(directory + "critic_" + str(critic_id) + ".pt")

        for target_critic_id, target_critic in enumerate(self.target_critics):
            self.target_critics[target_critic_id] = \
                torch.load(directory + "target_critic_" + str(target_critic_id) + ".pt")

        self.actor = torch.load(directory + "actor.pt")
        self.target_actor = torch.load(directory + "target_actor.pt")

        with open(directory + "replay_buffer.pkl", "rb") as f:
            self.replay_buffer = pickle.load(f)
