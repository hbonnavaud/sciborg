from random import randrange
from typing import Union
from gymnasium.spaces import Box, Discrete
from conditioned_wrapper import ConditioningWrapper
from ..value_based_agents.value_based_agent import ValueBasedAgent


class HER(ConditioningWrapper):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    NAME = "HER"

    def __init__(self,
                 reinforcement_learning_agent_class,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 goal_space: Union[Box, Discrete] = None,
                 goal_from_observation_fun=None,
                 nb_resample_per_observations: int = 4,
                 **params):

        # Super class init
        super().__init__(reinforcement_learning_agent_class, observation_space, action_space, =goal_space,
                         goal_from_observation_fun=goal_from_observation_fun, **params)

        # Trajectory relabelling attributes
        self.last_trajectory = []
        self.nb_resample_per_observations = nb_resample_per_observations

        # Modify instance name
        self.name = self.reinforcement_learning_agent.name + " + HER"

    def start_episode(self, *information, test_episode=False):
        """
        Args:
            observation: The first observation of the episode.
            conditioning: The conditioning of the episode.
            test_episode: Boolean indication whether the episode is a test episode or not.
            If it is a test episode, the agent will not explore (fully deterministic actions) and not learn (no
            interaction data storage or learning process).
        """
        observation, conditioning = information
        self.last_trajectory = []
        return super().start_episode(observation, goal, test_episode=test_episode)

    def process_interaction(self, action, next_observation, reward, done, learn=True):
        if learn and not self.under_test:
            self.last_trajectory.append((self.last_observation, action))
        super().process_interaction(action, next_observation, reward, done, learn=learn)

    def stop_episode(self):
        # Relabel last trajectory
        if self.under_test or len(self.last_trajectory) <= self.nb_resample_per_observations:
            return

        # For each observation seen :
        for observation_index, (observation, action) in enumerate(self.last_trajectory[:-self.nb_resample_per_observations]):
            new_observation_index = observation_index + 1
            new_observation, _ = self.last_trajectory[new_observation_index]

            # sample four goals in future observations
            for relabelling_id in range(self.nb_resample_per_observations):
                goal_index = randrange(new_observation_index, len(self.last_trajectory))
                target_observation, _ = self.last_trajectory[goal_index]
                goal = self.goal_from_observation_fun(target_observation)

                features = self.get_features(observation, goal)
                # Compute a reward that goes from -1, for the first observation of the fake trajectory, to 0, if the
                # new_observation if the fake goal.
                reward = 0 if new_observation_index == goal_index else -1
                new_features = self.get_features(new_observation, goal)

                done = goal_index == new_observation_index

                self.reinforcement_learning_agent.replay_buffer.append((features, action, reward, new_features, done))
        super().stop_episode()

    def goal_from_observation_fun(self, observation):
        return observation[..., :self.goal_space.shape[-1]]
