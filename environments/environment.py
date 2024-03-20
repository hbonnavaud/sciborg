from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Environment(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        :return: observation
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
    def copy(self):
        pass
