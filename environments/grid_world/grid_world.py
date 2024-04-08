import copy
import importlib
import math
import os
import random
import sys
from queue import PriorityQueue
from typing import Any, Tuple, Union, Dict, Optional
from gym import spaces
import numpy as np
from PIL import Image
from scipy.spatial import distance
from .utils.indexes import *
from .maps.maps_index import MapsIndex
from skimage.draw import line_aa
from utils import create_dir
from environments.environment import Environment


# un environment custom simple
def euclidean_distance(coordinates_1: tuple, coordinates_2: tuple) -> float:
    x1, y1 = coordinates_1
    x2, y2 = coordinates_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class GridWorld(Environment):
    name = "Grid-World"
    start_coordinates: Tuple[Union[int, Any], int]
    metadata = {'render.modes': ['human']}

    def __init__(self, map_name: str = MapsIndex.EMPTY.value):
        grid_world_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, grid_world_path)
        self.map_name = map_name
        self.maze_map = np.array(importlib.import_module("grid_world.maps." + self.map_name).maze_array)
        self.height, self.width = self.maze_map.shape
        self.agent_coordinates = None

        low = np.array([.5 - self.width / 2, - (self.height / 2 - .5)])
        high = np.array([self.width / 2 - .5, - (.5 - self.height / 2)])
        self.observation_space = spaces.Box(low=np.float32(low), high=np.float32(high))
        self.action_space = spaces.Discrete(len(Direction))
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

    def is_valid_coordinates(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_valid_observation(self, observation):
        return self.is_valid_coordinates(*self.get_coordinates(observation))

    def get_tile_type(self, x, y):
        return TileType(self.maze_map[y][x].item())

    def is_terminal_tile(self, x, y):
        observation_type = self.get_tile_type(x, y)
        return observation_type == TileType.TERMINAL

    def is_available(self, x, y):
        # False for 218, 138
        # if we move into a row not in the grid
        if 0 > x or x >= self.width or 0 > y or y >= self.height:
            return False
        if self.get_tile_type(x, y) == TileType.WALL:
            return False
        return True

    def get_new_coordinates(self, action):
        agent_x, agent_y = self.agent_coordinates
        if Direction(action) == Direction.TOP:
            agent_y -= 1
        elif Direction(action) == Direction.BOTTOM:
            agent_y += 1
        elif Direction(action) == Direction.LEFT:
            agent_x -= 1
        elif Direction(action) == Direction.RIGHT:
            agent_x += 1
        else:
            raise AttributeError("Unknown action")
        return agent_x, agent_y

    def step(self, action):
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            done = self.is_terminal_tile(new_x, new_y)
            reward = -1 if not done else 1
            self.agent_coordinates = new_x, new_y
            return self.get_observation(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done, {}
        else:
            return self.get_observation(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False, {}

    def reset(self):
        self.agent_coordinates = np.flip(random.choice(np.argwhere(self.maze_map == 2)))
        observation = self.get_observation(*self.agent_coordinates)
        return observation

    def get_color(self, x, y, ignore_agent=False, ignore_terminals=False):
        agent_x, agent_y = self.agent_coordinates
        if (agent_x, agent_y) == (x, y) and not ignore_agent:
            return Colors.AGENT.value
        else:
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
                cell_color = self.get_color(x, y, ignore_agent=ignore_agent, ignore_terminals=ignore_rewards)
                img = self.set_tile_color(img, x, y, cell_color)
        return img

    def get_oracle(self) -> list:
        """
        Return an oracle as a list of every possible observations inside the environment.
        """
        reachable_coordinates = np.argwhere(self.maze_map != 1).tolist()
        return [self.get_observation(x, y) for x, y in reachable_coordinates]

    def render(self, ignore_rewards=False):
        """
        Render the whole-grid human view
        """

        img = self.get_environment_background(ignore_agent=False, ignore_rewards=ignore_rewards)
        agent_x, agent_y = self.agent_coordinates
        self.place_point(img, self.get_observation(agent_x, agent_y), Colors.AGENT.value)
        return img

    def get_available_positions(self, coordinates: tuple) -> list:
        """
        return an list of every available coordinates from the given one (used for A*).
        """
        x, y = coordinates  # Make sure coordinates is a tuple

        available_coordinates = []
        if x < (self.width - 1):
            new_coord = (x + 1, y)
            if self.is_available(x + 1, y):
                available_coordinates.append((new_coord, Direction.RIGHT.value))
        if x > 0:
            new_coord = (x - 1, y)
            if self.is_available(x - 1, y):
                available_coordinates.append((new_coord, Direction.LEFT.value))

        if y < (self.height - 1):
            new_coord = (x, y + 1)
            if self.is_available(x, y + 1):
                available_coordinates.append((new_coord, Direction.BOTTOM.value))
        if y > 0:
            new_coord = (x, y - 1)
            if self.is_available(x, y - 1):
                available_coordinates.append((new_coord, Direction.TOP.value))

        return available_coordinates

    def best_path(self, observation_1, observation_2):
        """
        Return the shortest distance between two tiles, in number of action the agent needs to go from one to another.
        :param observation_1: Start observation,
        :param observation_2: Destination observation,
        :return: Shortest path (using A*), as a list of action to move from observation_1 to observation_2.
        """
        # Remove trivial case
        if isinstance(observation_1, tuple):
            coordinates_1 = observation_1
        else:
            coordinates_1 = self.get_coordinates(observation_1)
        if isinstance(observation_2, tuple):
            coordinates_2 = observation_2
        else:
            coordinates_2 = self.get_coordinates(observation_2)

        frontier = PriorityQueue()
        frontier.put((0, coordinates_1))
        came_from: Dict[tuple, Optional[tuple]] = {}
        cost_so_far = {coordinates_1: 0}
        came_from[coordinates_1] = None

        while not frontier.empty():
            priority, current = frontier.get()

            if current == coordinates_2:
                break

            for next_position, action in self.get_available_positions(current):
                new_cost = cost_so_far[current] + 1
                if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
                    cost_so_far[next_position] = new_cost
                    priority = new_cost + euclidean_distance(next_position, coordinates_2)
                    frontier.put((priority, next_position))
                    came_from[next_position] = current

        return came_from, cost_so_far

    def distance(self, observation_1, observation_2):
        _, distance = self.best_path(observation_1, observation_2)
        coordinates_2 = self.get_coordinates(observation_2)
        return distance[coordinates_2]

    def show_path_on_image(self, observation_1, observation_2, file_directory, file_name, colors) -> None:
        """
        Save an image of the environment with many path draw on it
        :param observation_1: start observation,
        :param observation_2: destination observation,
        :param file_directory: destination where the image should be saved
        :param file_name: name of the future image (without .png)
        :param colors: list of colors of shape [[R1, G1, B1], [R1, G2, B2], ... ] and len = len(paths)
        """

        best_path = self.best_path(observation_1, observation_2)

        if isinstance(observation_1, tuple):
            coordinates_1 = observation_1
        else:
            coordinates_1 = self.get_coordinates(observation_1)
        if isinstance(observation_2, tuple):
            coordinates_2 = observation_2
        else:
            coordinates_2 = self.get_coordinates(observation_2)

        if not self.is_available(coordinates_1[0], coordinates_1[1]) or not self.is_available(coordinates_2[0],
                                                                                              coordinates_2[1]):
            print("one of these observations is not available")

        # Generate image
        image = self.get_environment_background()

        # Draw this path
        for coordinates in best_path.coordinates:
            tile_x, tile_y = coordinates
            self.set_tile_color(image, tile_x, tile_y, colors)

        # Save image
        image = Image.fromarray(image)
        create_dir(file_directory)
        if not file_name.endswith(".png"):
            if len(file_name.split(".")) > 1:
                file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
            file_name += ".png"
        image.save(file_directory + file_name)

    def sample_reachable_observation(self):
        observation_coordinates = np.flip(random.choice(np.argwhere(self.maze_map != 1)))
        return self.get_observation(*observation_coordinates)

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
        radius = round(width / 2) + 1
        for i in range(center_x - radius, center_x + radius):
            for j in range(center_y - radius, center_y + radius):
                if distance.euclidean((i + 0.5, j + 0.5), (center_x, center_y)) < radius:
                    if 0 <= j < image.shape[0] and 0 <= i < image.shape[1]:
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

        color = np.array(color) if isinstance(color, list) else color
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

        return image
