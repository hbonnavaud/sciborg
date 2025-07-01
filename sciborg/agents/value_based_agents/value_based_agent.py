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

    def scale_action(self, actions: Union[np.ndarray, torch.Tensor], source_action_box: Box):
        """
        Scale an action within the given bounds action_low to action_high, to our action_space.
        The result action is also clipped to fit in the action space in case the given action wasn't exactly inside
        the given bounds.
        Useless if our action space is discrete.
        @return: scaled and clipped actions. WARNING: actions are both attribute and result. They are modified by the
        function. They are also returned for better convenience.
        """
        assert isinstance(self.action_space, Box), \
            "Scale_action is useless and cannot work if our action space is discrete."
        assert isinstance(actions, np.ndarray) or isinstance(actions, torch.Tensor)
        assert isinstance(source_action_box, Box)

        source_low, source_high = source_action_box.low, source_action_box.high
        target_low, target_high = self.action_space.low, self.action_space.high
        if isinstance(actions, torch.Tensor):
            source_low, source_high = (torch.tensor(source_low, device=self.device),
                                       torch.tensor(source_high, device=self.device))
            target_low, target_high = (torch.tensor(target_low, device=self.device),
                                       torch.tensor(target_high, device=self.device))

        # Scale action to the action space
        scale = (target_high - target_low) / (source_high - source_low)
        actions = actions * scale + (target_low - (source_low * scale))

        # Clip actions to the action space to prevent floating point errors
        clip_fun = np.clip if isinstance(actions, np.ndarray) else torch.clamp
        return clip_fun(actions, target_low, target_high)

    @abstractmethod
    def get_value(self, features, actions=None):
        return 0

    @abstractmethod
    def learn(self):
        pass
