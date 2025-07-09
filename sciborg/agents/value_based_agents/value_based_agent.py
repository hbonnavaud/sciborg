from typing import Union
import numpy as np
import torch

from ..rl_agent import RLAgent
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete


class ValueBasedAgent(RLAgent, ABC):

    def __init__(self, *args, **params): 
        """
        @param observation_space: Environment's observation space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """
        super().__init__(*args, **params)

    @abstractmethod
    def get_value(self, observations: np.ndarray, actions: np.ndarray = None):
        """
        Args:
            observations: the observation(s) from which we want to obtain a value. Could be a batch.
            actions: the action that will be performed from the given observation(s). If none, the agent compute itself
                which action it would have taken from these observations.
        Returns: the value of the given features.
        """
        raise NotImplementedError("This function is not implemented at the interface level.")
