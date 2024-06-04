import copy
from typing import Union

import numpy as np
import importlib
import random
from gym import spaces
from matplotlib import pyplot as plt
from .utils.indexes import *
from .maps.maps_index import MapsIndex
from scipy.spatial import distance
from skimage.draw import line_aa
from ..environment import Environment


class PointEnv(Environment):
    """Abstract class for 2D navigation environments."""

    name = "Point-Maze"

    def __init__(self, map_name: str = MapsIndex.EMPTY.value, action_noise=1.0, reset_anywhere=True):
        self.reset_anywhere = reset_anywhere
        self.maze_map = np.array(importlib.import_module("sciborg.environments.point_maze.maps." + map_name).maze_array)
        self.height, self.width = self.maze_map.shape
        self.action_noise = action_noise
        self.observation_space = spaces.Box(low=np.array([- self.width / 2, - self.height / 2]).astype(np.float32),
                                            high=np.array([self.width / 2, self.height / 2]).astype(np.float32))
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]).astype(np.float32),
                                       high=np.array([1.0, 1.0]).astype(np.float32), dtype=np.float32)
        self.agent_observation = None
        self.reset()

    def get_observation(self, x, y):
        """
        Return a numpy array (observation) that belongs to X and Y coordinates in the grid.
        """
        x_value = x + .5 - self.width / 2
        y_value = - (y + .5 - self.height / 2)
        return np.asarray([x_value, y_value])

    def get_coordinates(self, observation):
        return round(observation[0].item() -.5 + self.width / 2), round(- observation[1].item() - .5 + self.height / 2)

    def reset(self):
        if self.reset_anywhere:
            # Look for reachable tile in the map
            start_coordinates = np.flip(random.choice(np.argwhere(self.maze_map != 1)))
        else:
            # Look for a valid start position in the map
            start_coordinates = np.flip(random.choice(np.argwhere(self.maze_map == 2)))
        self.agent_observation = self.get_observation(*start_coordinates)
        if self.reset_anywhere:
            self.agent_observation += np.random.uniform(low=-0.5, high=0.5, size=2)
        assert self.observation_space.contains(self.agent_observation.astype(self.observation_space.dtype))
        return self.agent_observation.copy()

    def _sample_empty_observation(self):
        empty_coordinates = np.flip(random.choice(np.argwhere(self.maze_map != 1)))
        observation = self.get_observation(*empty_coordinates)
        observation += np.random.uniform(low=-0.5, high=0.5, size=2)
        assert self.is_available(*self.get_coordinates(observation))
        return observation

    def get_tile_type(self, x, y):
        return TileType(self.maze_map[y][x].item())

    def is_available(self, x, y):
        # False for 218, 138
        # if we move into a row not in the grid
        if 0 > x or x >= self.width or 0 > y or y >= self.height:
            return False
        if self.get_tile_type(x, y) == TileType.WALL:
            return False
        return True

    def step(self, action):
        assert self.action_space.contains(action.astype(self.action_space.dtype))
        if self.action_noise > 0:
            action += np.random.normal(0, self.action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        num_sub_steps = 10
        dt = 1.0 / num_sub_steps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_sub_steps):
            new_observation = self.agent_observation.copy()
            for axis in range(num_axis):
                new_observation[axis] += dt * action[axis]
            if self.is_available(*self.get_coordinates(new_observation)):
                self.agent_observation = new_observation

        done = False
        new_tile_type = self.get_tile_type(*self.get_coordinates(self.agent_observation))
        reward = 10 if new_tile_type == TileType.TERMINAL else -1
        return self.agent_observation.copy(), reward, done, {}

    def get_oracle(self, step=0.2):
        oracle = []
        observation_range = self.observation_space.high - self.observation_space.low
        x_range = int(observation_range[0].item() * (1 / step))
        y_range = int(observation_range[1].item() * (1 / step))
        for x in range(x_range):
            for y in range(y_range):
                observation = np.array([x * step + (step / 2) + self.observation_space.low[0].item(),
                                  y * step + (step / 2) + self.observation_space.low[1].item()])
                oracle.append(observation)
        return oracle

    """
    RENDERING FUNCTIONS
    """

    def set_tile_color(self, image_array: np.ndarray, x, y, color, tile_size=10, border_size=0) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param x: X coordinate of the tile to set
        :param y: Y coordinate of the tile to set
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :param tile_size: size of the tile in pixels
        :param border_size: size of the tile's border in pixels
        :return: The new image
        """
        tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

        if border_size > 0:
            tile_img[:, :, :] = Colors.TILE_BORDER.value
            tile_img[border_size:-border_size, border_size:-border_size, :] = color
        else:
            tile_img[:, :, :] = color

        y_min = y * tile_size
        y_max = (y + 1) * tile_size
        x_min = x * tile_size
        x_max = (x + 1) * tile_size
        image_array[y_min:y_max, x_min:x_max, :] = tile_img
        return image_array

    def get_color(self, x, y, ignore_terminals=False):
        tile_type = self.get_tile_type(x, y)
        if tile_type == TileType.START:
            return Colors.START.value
        elif tile_type == TileType.WALL:
            return Colors.WALL.value
        elif tile_type == TileType.EMPTY:
            return Colors.EMPTY.value
        elif tile_type == TileType.TERMINAL:
            return Colors.EMPTY.value if ignore_terminals else Colors.TERMINAL.value
        else:
            raise AttributeError("Unknown tile type")

    def get_environment_background(self, tile_size=10, ignore_agent=True, ignore_rewards=False) -> np.ndarray:
        """
        Return an image (as a numpy array of pixels) of the environment background.
        :return: environment background -> np.ndarray
        """
        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for y in range(self.height):
            for x in range(self.width):
                cell_color = self.get_color(x, y, ignore_terminals=ignore_rewards)
                img = self.set_tile_color(img, x, y, cell_color)

        if not ignore_agent:
            self.place_point(img, self.agent_observation, Colors.AGENT.value, 5)

        return img

    def render(self, ignore_rewards=False):
        """
        Render the whole-grid human view
        """
        img = self.get_environment_background(ignore_agent=False, ignore_rewards=ignore_rewards)
        return img

    def place_point(self, image: np.ndarray, observation, color: Union[np.ndarray, list], width=5):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the observation space of the point to place.
        param y: y coordinate in the observation space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        if isinstance(color, list):
            color = np.array(color)

        observation_space_range = (self.observation_space.high - self.observation_space.low)
        center = (observation - self.observation_space.low) / observation_space_range
        center[1] = 1 - center[1]
        center_y, center_x = (image.shape[:2] * np.flip(center)).astype(int)

        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        radius = width
        for i in range(center_x - radius, center_x + radius):
            for j in range(center_y - radius, center_y + radius):
                dist = distance.euclidean((i, j), (center_x, center_y))
                if dist < radius and 0 <= i < image.shape[1] and 0 <= j < image.shape[0]:
                    image[j, i] = color
        return image

    def place_edge(self, image: np.ndarray, observation_1, observation_2, color: Union[np.ndarray, list], width=40):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the observation space of the point to place.
        param y: y coordinate in the observation space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        if isinstance(color, list):
            color = np.array(color)
        observation_space_range = (self.observation_space.high - self.observation_space.low)

        center = (observation_1 - self.observation_space.low) / observation_space_range
        center[1] = 1 - center[1]
        center_y_1, center_x_1 = (image.shape[:2] * np.flip(center)).astype(int)

        center = (observation_2 - self.observation_space.low) / observation_space_range
        center[1] = 1 - center[1]
        center_y_2, center_x_2 = (image.shape[:2] * np.flip(center)).astype(int)

        rr, cc, val = line_aa(center_y_1, center_x_1, center_y_2, center_x_2)
        old = image[rr, cc]
        extended_val = np.tile(val, (3, 1)).T
        image[rr, cc] = (1 - extended_val) * old + extended_val * color

    def plot(self):
        """
        plot the environment in matplotlib windows
        """
        walls = self.maze_map.T
        (height, width) = walls.shape
        for (i, j) in zip(*np.where(walls)):
            x = np.array([j, j + 1]) / float(width)
            y0 = np.array([i, i]) / float(height)
            y1 = np.array([i + 1, i + 1]) / float(height)
            plt.fill_between(x, y0, y1, color='grey')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([])
        plt.yticks([])

    def copy(self):
        return copy.deepcopy(self)
