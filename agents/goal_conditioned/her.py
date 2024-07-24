# Goal conditioned agent
from random import randrange
from typing import Union

import numpy as np
from gym.spaces import Box, Discrete

from .goal_conditioned_wrapper import GoalConditionedWrapper
from ..value_based_agents.value_based_agent import ValueBasedAgent


class HER(GoalConditionedWrapper):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self,
                 reinforcement_learning_agent_class,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 goal_space: Union[Box, Discrete] = None,
                 goal_from_observation_fun=None,
                 nb_resample_per_observations: int = 4,
                 **params):

        # Super class init
        super().__init__(reinforcement_learning_agent_class, observation_space, action_space, goal_space=goal_space,
                         goal_from_observation_fun=goal_from_observation_fun, **params)

        # Trajectory relabelling attributes
        self.last_trajectory = []
        self.nb_resample_per_observations = nb_resample_per_observations

        # Modify instance name
        self.name = self.reinforcement_learning_agent.name + " + HER"

    def start_episode(self, *information, test_episode=False):
        observation, goal = information
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
            next_observation_index = observation_index + 1
            next_observation, _ = self.last_trajectory[next_observation_index]

            # sample four goals in future observations
            for relabelling_id in range(self.nb_resample_per_observations):
                goal_index = randrange(next_observation_index, len(self.last_trajectory))
                target_observation, _ = self.last_trajectory[goal_index]
                goal = self.goal_from_observation_fun(target_observation)

                features = self.get_features(observation, goal)
                # Compute a reward that goes from -1, for the first observation of the fake trajectory, to 0, if the
                # next_observation if the fake goal.
                reward = (next_observation_index / goal_index) - 1
                next_features = self.get_features(next_observation, goal)

                done = goal_index == next_observation_index

                (self.reinforcement_learning_agent.replay_buffer
                 .append((features, action, reward, next_features, done)))
        super().stop_episode()
