from typing import Union
import numpy as np
import torch

from ..rl_agent import RLAgent
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Discrete


class ValueBasedAgent(RLAgent, ABC):

    def __init__(self, *args, **params): 
        """
        Args:
            observation_space: Environment's observation space.
            action_space: Environment's action_space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
        """
        super().__init__(*args, **params)

    @abstractmethod
    def get_value(self, observations: np.ndarray, actions: np.ndarray = None) -> np.ndarray:
        """
        Args:
            observations (np.ndarray): The observation(s) from which we want to obtain a value. Could be a batch.
            actions (np.ndarray, optional): The action that will be performed from the given observation(s). If none,
                the agent compute itself which action it would have taken from these observations.
        Returns:
            np.ndarray: The value of the given features.
        """
        raise NotImplementedError("This function is not implemented at the interface level.")
