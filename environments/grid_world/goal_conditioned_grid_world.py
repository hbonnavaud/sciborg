import copy
import random
import numpy as np
from .grid_world import GridWorld
from .maps.maps_index import MapsIndex
from .utils.indexes import Colors
from ..goal_conditioned_environment import GoalConditionedEnvironment


class GoalConditionedGridWorld(GridWorld, GoalConditionedEnvironment):

    name = "Goal-conditioned Grid-World"

    def __init__(self, map_name: str = MapsIndex.EMPTY.value):
        super().__init__(map_name=map_name)
        self.goal_space = copy.deepcopy(self.observation_space)
        self.goal_coordinates = None
        self.goal = None
        self.reachability_threshold = 0.1  # In this environment, an (s - g) L2 norm below this threshold implies s = g

    def reset_goal(self):
        """
        Choose a goal for the agent.
        """
        self.goal_coordinates = np.flip(random.choice(np.argwhere(self.maze_map != 1)))
        self.goal = self.get_observation(*self.goal_coordinates)

    def goal_reached(self):
        """
        Return a boolean True if the agent observation is on the goal (and exactly on the goal since our observation space is
        discrete here in reality), and false otherwise.
        """
        return (self.agent_coordinates == self.goal_coordinates).all()

    def step(self, action):
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            self.agent_coordinates = new_x, new_y
            done = self.goal_reached()
            reward = -1 if not done else 0
            return self.get_observation(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done, {"reached": done}
        else:
            return self.get_observation(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False, {"reached": False}

    def reset(self) -> tuple:
        """
        Return the initial observation, and the selected goal.
        """
        self.reset_goal()
        return super().reset(), self.goal

    def reached(self):
        raise NotImplementedError()

    def get_goal_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.copy()

    def render(self, ignore_rewards=True, ignore_goal=False):
        """
        Render the whole-grid human view (get view from super class then add the goal over the image)
        """
        img = super().render(ignore_rewards=ignore_rewards)
        if not ignore_goal:
            goal_x, goal_y = self.goal_coordinates
            self.place_point(img, self.get_observation(goal_x, goal_y), Colors.GOAL.value)
        return img
