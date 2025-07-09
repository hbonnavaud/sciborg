from agents.sac.continuous.multi_actions_actor import ActorNetwork
from agents.sac.continuous.sac_agent import SACAgent
from agents.skills_learner import SkillLearner
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces import Box
import torch

from agents.utils.default_nn import DefaultNN
from utils.general_fun import create_dir, empty_dir


class DiscriminatorReplayBuffer:
    def __init__(self, state_dim, max_size=3000):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.skills_memory = np.zeros(self.mem_size, dtype=np.int)

        # Iterator_parameters
        self.iterator_pointer = 0

    def store_experience(self, state, skill, index=None):
        if index is None:
            index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.skills_memory[index] = skill

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        skills = self.skills_memory[batch]

        return states, skills

    def __iter__(self):
        self.iterator_pointer = 0
        return self

    def __next__(self) -> tuple:
        if self.iterator_pointer < min(self.mem_counter, self.mem_size):
            self.iterator_pointer += 1
            return self.state_memory[self.iterator_pointer - 1], self.skills_memory[self.iterator_pointer - 1]
        else:
            raise StopIteration

    def is_full(self):
        return self.mem_counter >= self.mem_size


class Discriminator(DefaultNN):

    def __init__(self, learning_rate, state_size, nb_skills, layer1_size=256, layer2_size=256, tau=.1):
        super().__init__(learning_rate=learning_rate, input_size=state_size, output_size=nb_skills,
                         layer1_size=layer1_size, layer2_size=layer2_size, tau=tau)

    def sample_categorical(self, state):
        values = self.forward(state)
        result_probabilities = torch.softmax(values, 0)

        distribution = torch.distributions.Categorical(result_probabilities)
        skill_index = distribution.sample()
        log_probabilities = distribution.log_prob(skill_index)
        entropy = distribution.entropy()

        return skill_index, log_probabilities, entropy, result_probabilities

    def __copy__(self):
        new_one = Discriminator(self.learning_rate, self.input_size, self.output_size, layer1_size=self.layer1_size,
                                layer2_size=self.layer2_size, tau=self.tau)
        new_one.update_parameters_following(self, tau=1)
        return new_one


class DIAYNAgentContinuous(SACAgent, SkillLearner):

    def __init__(self, state_space, action_space, nb_skills, environment=None,
                 # Learning rates
                 actor_lr=0.003,
                 critic_lr=0.003,
                 disc_lr=0.003,
                 # Networks parameters
                 rl_alg_layer1_size=128,
                 rl_alg_layer2_size=64,
                 discriminator_layer1_size=128,
                 discriminator_layer2_size=64,
                 # Buffers and batches parameters
                 rl_alg_max_buffer_size=10000,
                 discriminator_max_buffer_size=1000,
                 rl_alg_batch_size=128,
                 discriminator_batch_size=128,
                 # Other Hyper-parameters
                 sac_temperature=1,
                 gamma=0.99,
                 tau=0.005):
        """
            ** SkillLearningAgent characteristic **
             - The embedded agent will not use the stage given by the environment, but an intrinsically build state
        """
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.disc_lr = disc_lr
        self.rl_alg_layer1_size = rl_alg_layer1_size
        self.rl_alg_layer2_size = rl_alg_layer2_size
        self.discriminator_layer1_size = discriminator_layer1_size
        self.discriminator_layer2_size = discriminator_layer2_size
        self.rl_alg_max_buffer_size = rl_alg_max_buffer_size
        self.rl_alg_batch_size = rl_alg_batch_size
        self.sac_temperature = sac_temperature

        self.state_shape = state_space.shape
        self.extrinsic_state_size = self.state_shape[0] if isinstance(state_space, Box) else state_space.n

        # Initialise super class, this one will use extrinsic informations,
        SkillLearner.__init__(self, nb_skills)
        SACAgent.__init__(self, self.get_intrinsic_state_space(), action_space, environment=environment,
                          actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, max_buffer_size=rl_alg_max_buffer_size,
                          tau=tau, layer1_size=rl_alg_layer1_size, layer2_size=rl_alg_layer2_size,
                          batch_size=rl_alg_batch_size, alpha=sac_temperature)
        self.intrinsic_state_size = self.state_size

        # Discriminator replay buffer initialisation
        self.discriminator_max_buffer_size = discriminator_max_buffer_size
        self.discriminator_memory = DiscriminatorReplayBuffer(self.extrinsic_state_size,
                                                              max_size=self.discriminator_max_buffer_size)
        self.discriminator_batch_size = discriminator_batch_size
        self.discriminator = Discriminator(learning_rate=disc_lr, state_size=self.extrinsic_state_size,
                                           nb_skills=self.nb_skills, layer1_size=discriminator_layer1_size,
                                           layer2_size=discriminator_layer2_size)

        self.old_actor = ActorNetwork(actor_lr, self.intrinsic_state_size, nb_actions=self.nb_actions,
                                      actions_bounds=(self.lower_bound, self.higher_bound))
        self.old_discriminator = Discriminator(learning_rate=disc_lr, state_size=self.extrinsic_state_size,
                                               nb_skills=self.nb_skills, layer1_size=discriminator_layer1_size,
                                               layer2_size=discriminator_layer2_size)

        self.checkpoints_dir = "outputs_1.0_noise/checkpoints/"

    def set_outputs_directory(self, outputs_directory):
        SkillLearner.set_outputs_directory(self, outputs_directory)
        self.checkpoints_dir = outputs_directory + "checkpoints/"

    def get_intrinsic_state_space(self):
        intrinsic_state_space = Discrete(self.extrinsic_state_size + self.nb_skills)
        return intrinsic_state_space

    def get_intrinsic_state(self, extrinsic_state, skill_index=None):
        if skill_index is None:
            skill_index = self.skill_index
        assert 0 <= skill_index < self.nb_skills
        extrinsic_state = self.preprocess_state(extrinsic_state)
        skill_state = np.zeros(self.nb_skills)
        skill_state[skill_index] = 1.
        intrinsic_state = np.concatenate((extrinsic_state, skill_state))
        intrinsic_state = torch.Tensor(intrinsic_state).to(self.device)
        return intrinsic_state

    def compute_skill(self, state):
        computed_skill = self.discriminator.forward(state)
        computed_skill = torch.max(computed_skill)
        return computed_skill

    def get_intrinsic_reward(self, new_state, skill_index=None, discriminator_network=None,
                             get_discriminator_entropy=False):
        output_entropy = 0
        if skill_index is None:
            skill_index = self.skill_index
        assert 0 <= skill_index < self.nb_skills
        if discriminator_network is None:
            discriminator_network = self.discriminator
        assert isinstance(discriminator_network, Discriminator)
        with torch.no_grad():
            computed_skill = discriminator_network.forward(new_state)
            if get_discriminator_entropy:
                distribution = torch.distributions.Categorical(torch.nn.Softmax(dim=0)(computed_skill))
                output_entropy = distribution.entropy().item()
            computed_skill = computed_skill.reshape((1, len(computed_skill)))
            skill = torch.tensor([skill_index], dtype=torch.int64).to(self.device)
            intrinsic_reward = - torch.nn.CrossEntropyLoss()(computed_skill, skill)
            intrinsic_reward -= self.p_z.log_prob(torch.tensor(skill_index).to(self.device))
        if get_discriminator_entropy:
            return intrinsic_reward.item(), output_entropy
        return intrinsic_reward.item()

    def action(self, state, skill_index=None, actor_network=None):
        if skill_index is None:
            skill_index = self.skill_index
        assert 0 <= skill_index < self.nb_skills
        if actor_network is None:
            actor_network = self.actor
        assert isinstance(actor_network, ActorNetwork)
        state = self.preprocess_state(state)
        state = self.get_intrinsic_state(state, skill_index=skill_index)
        actions, _, _ = actor_network.sample_normal(state, reparameterize=False)

        return actions.detach().cpu().numpy()

    def on_action_stop(self, action, new_state, reward, done):
        intrinsic_last_state = self.get_intrinsic_state(self.last_state)
        intrinsic_new_state = self.get_intrinsic_state(new_state)
        intrinsic_reward = self.get_intrinsic_reward(new_state)
        SACAgent.remember(self, intrinsic_last_state, action, intrinsic_reward, intrinsic_new_state, done)

        new_state = self.preprocess_state(new_state)
        self.discriminator_memory.store_experience(new_state, self.skill_index)

        discriminator_error = self.diayn_learn()
        self.last_state = new_state
        return intrinsic_reward, discriminator_error

    def on_episode_start(self, state, episode_id):
        """ Function called when a new episode is started """
        intrinsic_state = self.get_intrinsic_state(state)
        SACAgent.on_episode_start(self, intrinsic_state, episode_id)
        self.last_state = state

    def get_discriminator_batch(self):
        if self.discriminator_memory.mem_counter < self.discriminator_batch_size:
            raise Exception

        states_batch, skills_batch = self.discriminator_memory.sample_buffer(self.discriminator_batch_size)

        states_batch = torch.tensor(states_batch, dtype=torch.float).to(self.device)
        skills_batch = torch.tensor(skills_batch, dtype=torch.int64).to(self.device)
        return states_batch, skills_batch

    def diayn_learn(self):
        SACAgent.learn(self)

        """ Apprentissage de l'agent apprenant les compétences, pas de l'agent intégré. """

        if self.discriminator_memory.mem_counter < self.discriminator_batch_size:
            return
        try:
            states_batch, skills_batch = self.get_discriminator_batch()
        except:
            return

        discriminator_loss_function = torch.nn.CrossEntropyLoss()

        skill_predictions_batch = self.discriminator.forward(states_batch)

        self.discriminator.optimizer.zero_grad()
        discriminator_loss = discriminator_loss_function(skill_predictions_batch, skills_batch)
        discriminator_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
        self.discriminator.optimizer.step()
        return discriminator_loss.item()

    # SKILL LEARNER METHODS

    def save_learning(self, skills):
        """
        Save modules that learned things in this sequence, and associate it with the skills that was learned inside it.
        :param skills: list of skills learned during this sequence (aka. learning phase) that wil be associated with
        modules that learned on it.
        """

        create_dir(self.checkpoints_dir + "actor/")
        create_dir(self.checkpoints_dir + "discriminator/")
        empty_dir(self.checkpoints_dir + "actor/")
        empty_dir(self.checkpoints_dir + "discriminator/")

        for skill in skills:
            destination = self.checkpoints_dir + "actor/" + str(skill) + ".pth.tar"
            torch.save(self.actor.state_dict(), destination)
            destination = self.checkpoints_dir + "discriminator/" + str(skill) + ".pth.tar"
            torch.save(self.discriminator.state_dict(), destination)

    def get_test_action(self, state, sequence_id: int, skill_id: int, policy_tested: bool):
        """ Check Mother class (SkillLearner) implementation for detailed description """

        create_dir(self.checkpoints_dir + "actor/")
        create_dir(self.checkpoints_dir + "discriminator/")
        empty_dir(self.checkpoints_dir + "actor/")
        empty_dir(self.checkpoints_dir + "discriminator/")

        if policy_tested:
            return self.action(state, skill_index=skill_id)
        else:
            # Then we should find the perfect policy that was specifically trained on this skill
            destination = self.checkpoints_dir + "actor/" + str(skill_id) + ".pth.tar"
            try:
                self.old_actor.load_state_dict = torch.load(destination)
                return self.action(state, skill_index=skill_id, actor_network=self.old_actor)
            except:  # There's no old knowledge for the given skill, probably because it's one of the skill we are
                # actually learning.
                return self.action(state, skill_index=skill_id)

    def get_test_grade(self, path: list, sequence_id: int, skill_id: int, policy_tested: bool):
        """ Check Mother class (SkillLearner) implementation for detailed description """
        discriminator = self.discriminator
        if policy_tested:
            destination = self.checkpoints_dir + "discriminator/" + str(skill_id) + ".pth.tar"
            try:
                self.old_discriminator.load_state_dict = torch.load(destination)
                discriminator = self.old_discriminator
            except:  # There's no old knowledge for the given skill, probably because it's one of the skill we are
                # actually learning.
                pass

        # Now we can use this discriminator to evaluate our policy
        reward_sum = 0
        entropy_sum = 0
        for state in path:
            reward, entropy = self.get_intrinsic_reward(state, skill_index=skill_id,
                                                        discriminator_network=discriminator,
                                                        get_discriminator_entropy=True)
            reward_sum += reward
            entropy_sum += entropy
        return reward_sum / len(path), entropy_sum / len(path)

    def on_sequence_stop(self, sequence_id, learned_skills):
        """
        This function is called when a sequence stop.
        :param sequence_id: id of the sequence that stopped
        :param learned_skills: list of skills that has been learned during this sequence.
        """
        self.save_learning(learned_skills)
        self.discriminator_memory = DiscriminatorReplayBuffer(self.extrinsic_state_size,
                                                              max_size=self.discriminator_max_buffer_size)
        SACAgent.reset_buffer(self)

    def get_str_params(self):
        """
        Return a text with every informations about class parameters
        """

        return "\n\n=======================================================\n" + \
            " Agent DIAYNAgentDiscrete with parameters : \n" + \
            "# Learning rates\n" + \
            "actor_lr = " + str(self.actor_lr) + ",\n" + \
            "critic_lr = " + str(self.critic_lr) + ",\n" + \
            "disc_lr = " + str(self.disc_lr) + ",\n" + \
            "# Networks parameters\n" + \
            "rl_alg_layer1_size = " + str(self.rl_alg_layer1_size) + ",\n" + \
            "rl_alg_layer2_size = " + str(self.rl_alg_layer2_size) + ",\n" + \
            "discriminator_layer1_size = " + str(self.discriminator_layer1_size) + ",\n" + \
            "discriminator_layer2_size = " + str(self.discriminator_layer2_size) + ",\n" + \
            "# Buffers and batches parameters\n" + \
            "rl_alg_max_buffer_size = " + str(self.rl_alg_max_buffer_size) + ",\n" + \
            "discriminator_max_buffer_size = " + str(self.discriminator_max_buffer_size) + ",\n" + \
            "rl_alg_batch_size = " + str(self.rl_alg_batch_size) + ",\n" + \
            "discriminator_batch_size = " + str(self.discriminator_batch_size) + ",\n" + \
            "# Other Hyper-parameters\n" + \
            "sac_temperature = " + str(self.sac_temperature) + ",\n" + \
            "gamma = " + str(self.gamma) + ",\n" + \
            "tau = " + str(self.tau) + "\n" + \
            "=======================================================\n\n"