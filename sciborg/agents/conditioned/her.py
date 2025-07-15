from random import randrange
from typing import Union
from gymnasium.spaces import Box, Discrete
from .conditioning_wrapper import ConditioningWrapper
import numpy as np


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
                 nb_resample_per_observations: int = 4,
                 **params):
        """
        Args:
            observation_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Environment's observation space.
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Environment's action_space.
            conditioning_space (Union[gym.spaces.Box, gym.spaces.Discrete]): Conditioning space.
            name (str, optional): The agent's name.
            device (torch.device, optional): The device on which the agent operates.
            reinforcement_learning_agent_class: The class of the agent to be wrapped.
                This class will be instantiated using the given parameters, and the computed information (such as the
                new conditioned observation space).
        """

        # Super class init: goal_space is used as the ConditioningWrapper conditioning space, because our agent is
        # goal-conditioned here.
        super().__init__(reinforcement_learning_agent_class, observation_space, action_space, goal_space, **params)
        self.name = params.get("name", self.reinforcement_learning_agent.name + " + HER")

        # Trajectory relabelling attributes
        self.last_trajectory = []
        self.nb_resample_per_observations = nb_resample_per_observations

    def start_episode(self, *episode_information, test_episode: bool = False):
        """
        Args:
            episode_information (Tuple[np.ndarray, np.ndarray]): A tuple that contains:
                observation (np.ndarray): The first observation of the episode.
                goal (np.ndarray): The goal that will condition the episode.
            test_episode (bool, optional): Boolean indication whether the episode is a test episode or not.
            If it is a test episode, the agent will not explore (fully deterministic actions) and not learn (no
            interaction data storage or learning process).
        """
        observation, goal = episode_information
        self.last_trajectory = []
        return super().start_episode(observation, goal, test_episode=test_episode)

    def process_interaction(self,
                            action: np.ndarray,
                            reward: float,
                            new_observation: np.ndarray,
                            done: bool,
                            learn: bool = True):
        """
        Processed the passed interaction using the given information.
        The state from which the action has been performed is kept in the agent's attribute, and updated everytime this function is called.
        Therefore, it does not appear in the function signature.
        Args:
            action (np.ndarray): The action performed by the agent at this step.
            reward (float): The reward returned by the environment following this action.
            new_observation (np.ndarray): The new state reached by the agent with this action.
            done (bool): Whether the episode is done (no action will be performed from the given new_state) or not.
            learn (bool, optional): Whether the agent cal learn from this step or not (will define if the agent can
                save this interaction data, and start a learning step or not).
        """
        if learn and not self.under_test:
            self.last_trajectory.append((self.last_observation, action))
        super().process_interaction(action, reward, new_observation, done, learn=learn)

    def stop_episode(self):
        """
        Function that should be called everytime an episode is done.
        For most agents, it updates some variables, and fill the replay buffer with some more interaction data from the
        passed episode trajectory.
        """
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

    def goal_from_observation_fun(self, observation: np.ndarray) -> np.ndarray:
        """

        Args:
            observation (np.ndarray): The observation from which we want a goal.

        Returns:
            np.ndarray: A goal that could be associated to the given observation.
        """
        return observation[..., :self.goal_space.shape[-1]]
