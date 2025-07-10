from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from gymnasium.spaces import Box, Discrete
from ..rl_agent import RLAgent


class ConditionedAgent(RLAgent, ABC):
    """
    An interface defining the function that should be implemented for a conditioned agent.
    We call conditioned agent any RLAgent that have to choose action depending on the observation but also on a
    conditioning data. The said data could be a goal, a skill, or any information that could be added to the agent
    inputs.

    NB: You can ignore this class and condition the agent in your main function if you want.
    Those conditioned agent classes are made to implement some basic function such as concatenation between the
    conditioning data and the observation, at any step of the agent's life, in order to make yours easier.
    More, this interface make sure the conditioning wrappers (such as HER or DIAYN) respect the same functions
    signatures, and the base conditioning wrapper make their implementation easier.
    """

    def __init__(self,
                 observation_space: Union[Box, Discrete],
                 action_space: Union[Box, Discrete],
                 conditioning_space: Union[Box, Discrete],
                 **params):
        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Environment's action_space.
            conditioning_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Conditioning space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
        """

        super().__init__(observation_space=observation_space, action_space=action_space, **params)

        # Verify and store the conditioning space
        assert isinstance(conditioning_space, (Box, Discrete))
        self.init_params["conditioning_space"] = conditioning_space
        self.conditioning_space: Union[Box, Discrete] = conditioning_space

        # Compute the expected conditioning size
        if isinstance(self.conditioning_space, Box):
            self.conditioning_size = np.prod(self.observation_space.shape)
        else:
            self.conditioning_size = self.conditioning_space.n

        # The following variable keep in memory the conditioning associated with the current episode.
        # We consider that the agent will be conditioned by the same data during the entire episode, but this episode
        # conditioning could be overridden in the action function, so the agent can temporarily ignore its conditioning
        # and take an action conditioned by another data.
        self.current_conditioning = None

    def start_episode(self, *episode_information, test_episode=False) -> None:
        """
        Args:
            episode_information: a tuple containing two instances:
             - information[0]: observation: The first observation of the episode.
             - information[1]: conditioning: The conditioning of the episode.
            test_episode: Boolean indication whether the episode is a test episode or not.
            If it is a test episode, the agent will not explore (fully deterministic actions) and not learn (no
            interaction data storage or learning process).
        """
        observation, conditioning = episode_information
        assert self.observation_space.contains(observation)
        assert self.conditioning_space.contains(conditioning)
        super().start_episode(observation, test_episode=test_episode)

    @abstractmethod
    def action(self, observation, conditioning=None, explore=True) -> np.ndarray:
        """
        Args:
            observation: The observation from which we want the agent to take an action.
            conditioning: The data conditioning the agent for this action. If this data is None, then the conditioning
            used is the one stored fo the episode, i.e. self.current_conditioning.
            explore: Boolean indicating whether the agent can explore with this action of only exploit.
            If test_episode was set to True in the last self.start_episode call, the agent will exploit (explore=False)
            no matter the explore value here.
        Returns: The action chosen by the agent.
        """
        pass

    @abstractmethod
    def get_features(self, observations, conditioning=None) -> np.ndarray:
        """
        Args:
            observations: The observations from which we want the feature (Could be a single one or a batch).
            conditioning: The conditioning from which we want the feature (Could be a single one or a batch).
        Returns: the features that will be used as an input for the agent's NNs.
        """
        pass
