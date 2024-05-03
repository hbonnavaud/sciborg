import copy

import numpy as np
from .point_env import PointEnv
from .utils.indexes import Colors
from .maps.maps_index import MapsIndex
from ..goal_conditioned_environment import GoalConditionedEnvironment


class GoalConditionedPointEnv(PointEnv, GoalConditionedEnvironment):

    name = "Goal-conditioned Point-Maze"

    def __init__(self, map_name: str = MapsIndex.EMPTY.value, action_noise=1.0, reset_anywhere=True,
                 reachability_threshold=0.7, dense_reward=False):
        super().__init__(map_name=map_name, action_noise=action_noise, reset_anywhere=reset_anywhere)
        self.goal_space = copy.deepcopy(self.observation_space)
        self.goal = None
        self.reachability_threshold = reachability_threshold
        self.dense_reward = dense_reward

    def reset(self) -> tuple:
        """
        Return the initial observation, and the selected goal.
        """
        self.goal = self._sample_empty_observation().copy()
        return super().reset(), self.goal

    def reached(self, observation=None, goal=None):
        """
        Return True if the goal is considered as reached according to the environment reachability threshold.
        The observation used is the current agent observation if the observation parameter if left empty.
        The goal used id the current episode goal if the goal parameter is left empty.
        """
        observation = self.agent_observation.copy() if observation is None else observation
        goal = self.goal.copy() if goal is None else goal
        return np.linalg.norm(observation - goal) < self.reachability_threshold

    def step(self, action):
        super().step(action)

        done = self.reached()
        if self.dense_reward:
            reward = - np.linalg.norm(self.agent_observation - self.goal)
        else:
            reward = 0 if done else -1
        return self.agent_observation.copy(), reward, done, {"reached": done}

    def get_goal_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.copy()

    def render(self, ignore_rewards=True, ignore_goal=False):
        """
        Render the whole-grid human view (get view from super class then add the goal over the image)
        """
        img = super().render(ignore_rewards=ignore_rewards)
        if not ignore_goal:
            self.place_point(img, self.goal.copy(), Colors.GOAL.value)
        return img
