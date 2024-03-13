# Goal conditioned agent
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from gym.spaces import Box, Discrete
from agents.agent import Agent


class GoalConditionedAgent(Agent, ABC):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    name = "Default goal conditioned agent"

    def __init__(self, observation_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        Agent.__init__(self, observation_space, action_space, **params)
        self.current_goal = None

        # Compute out goal space
        self.goal_space = params.get("goal_space", self.observation_space)
        assert isinstance(self.goal_space, Box) or isinstance(self.goal_space, Discrete)
        self.goal_shape = self.goal_space.shape

