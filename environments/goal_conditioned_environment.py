from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class GoalConditionedEnvironment(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: observation, goal
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        :param action: action to perform in the environment
        :return: observation, reward, done, info
        """
        pass

    @abstractmethod
    def reached(self):
        """
        returns whether the agent has reached the goal
        """
        pass

    @abstractmethod
    def get_goal_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.copy()

    @abstractmethod
    def copy(self):
        pass
