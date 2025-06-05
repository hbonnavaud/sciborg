# Goal conditioned agent
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from gym.spaces import Box, Discrete
from ..rl_agent import RLAgent


class GoalConditionedAgent(RLAgent, ABC):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between RLAgent class and this
    one.
    """

    name = "Default goal conditioned agent"

    def __init__(self,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 goal_space=None,
                 **params
                 ):
        RLAgent.__init__(self, observation_space, action_space, **params)
        self.init_params["goal_space"] = goal_space
        self.current_goal = None

        # Compute out goal space
        self.goal_space = self.observation_space if goal_space is None else goal_space
        assert isinstance(self.goal_space, Box) or isinstance(self.goal_space, Discrete)
        self.goal_shape = self.goal_space.shape

    def start_episode(self, *episode_info, test_episode=False):
        observation, goal = episode_info
        super().start_episode(observation, test_episode=test_episode)
