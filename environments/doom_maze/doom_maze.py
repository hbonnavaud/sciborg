import importlib
import math
import os
import random
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Union, Tuple
from skimage.draw import line_aa
from scipy.spatial import distance
import numpy as np
from omg import WAD, MapEditor
from vizdoom import DoomGame, ScreenResolution, Mode, Gameobservation
from gym.spaces import Box, Discrete
from .maps.maps_index import MapsIndex
from .maps.tile_type import TileType

from ..goal_conditioned_environment import GoalConditionedEnvironment
from ...utils import generate_video, save_image
from .wads_builder import build_wad


class Colors(Enum):
    EMPTY = [250, 250, 250]
    WALL = [50, 54, 51]
    START = [213, 219, 214]
    TERMINAL = [73, 179, 101]
    TILE_BORDER = [50, 54, 51]
    AGENT = [0, 0, 255]
    GOAL = [255, 0, 0]


class DoomMaze(GoalConditionedEnvironment):

    name = "VizDoom-Maze"

    def __init__(self,
                 map_tag: MapsIndex = MapsIndex.MEDIUM,
                 reset_anywhere: bool = False,
                 random_walls_textures: bool = True,
                 sparse_reward: bool = True,
                 discrete_goals: bool = False,
                 goal_range: Union[Tuple[int, int], None] = None,
                 behaviour_file: str = "static_goal.acs",
                 configuration_file: str = "default.cfg",
                 block_size: int = 96,
                 attainability_threshold: int = 64,
                 pixels_per_tile: int = 20,
                 pixels_wall_width: int = 3,
                 action_frame_repeat: int = 4,
                 observation_resolution: ScreenResolution = ScreenResolution.RES_160X120):
        """
        Instantiate the class and build the .wad file.

        CONVENTION: There is two type of coordinates used in this file.
            - The coordinates of a tile inside the map_array (the array that describe the maze shape, a 2D list of
            TileType values (integers)). These coordinates are elements of a discrete space and are identified with
            h (for height) and w (width). self.map_array[h, w] return an int that represent the type of the tile at
            coordinates (h, w).
            The keyword "tile" in variable names refers to a tile in self.map_array so "tile_coordinates" refers to a
            coordinate (h, w) in the map_array.
            - The coordinates of a position inside the VizDoom maze, identified using x and y.
            These values are continuous.
             -> Functions self.get_tile_from_position and self.get_position_from_tile help doing the bridge from one to
             another.

        :param map_tag: element of DoomMapIndex enum that indicate from which map the .wad file should be built.
        :param reset_anywhere: If true, the agent position is sampled uniformly in the maze at each episode start.
            Otherwise, it is always set to one of the tiles equal to environments.maps.tile_typeTileType.START.value.
        :param random_walls_textures: Boolean, Indicate whether walls textures should be sampled randomly or not.
        :param sparse_reward: If true, the agent receive a reward of 0 when it reached the goal, -1 otherwise.
            If false, it will receive a reward equal to minus the Euclidean distance between its position and the goal
            one.
        :param discrete_goals: Boolean, Indicate whether goals should be sampled from a pre-defined set of fixed goals
            or uniformly in the entire maze
        :param goal_range: Union[Tuple[int, int], None]: Maximum distance allowed between the initial observation
            tile coordinates and the sampled goal position.
        :param behaviour_file: str: Path to the behaviour file. Should be a .cfg file. A default file is available for
            more information.
        :param configuration_file: str: Path to the configuration file. Should be a .acs file. A default file is
            available for more information.
        :param block_size: Size of a self.maze_map tile in the vizdoom maze (in pre-set maps, this is also the corridor
            width).
        :param attainability_threshold: We will consider a goal reached if the distance between the agent and the goal
            is bellow this threshold.
        :param pixels_per_tile: Only used for the self.render(mode="top_view"). Indicate the number of pixels for each
            self.maze_map tiles.
        :param pixels_wall_width: Only used for the self.render(mode="top_view"). Indicate the walls width in pixels
            when a wall is drawn.
        :param action_frame_repeat: How many times an action selected by the agent will be performed in the environment
            before we return the next observation, the reward, and the 'done' boolean.
        :param observation_resolution: Resolution (in pixels) of both the observation and the goal image given to the
            agent. Other values available: ScreenResolution.RES_320X240, ScreenResolution.RES_640X480.
        """
        assert isinstance(map_tag, MapsIndex), "Map tag should be an instance of .maps.maps_index.MapsIndex"
        assert isinstance(reset_anywhere, bool)
        assert isinstance(random_walls_textures, bool)
        assert isinstance(sparse_reward, bool)
        assert isinstance(discrete_goals, bool)
        assert not discrete_goals or goal_range is None, \
            "Impossible to ensure goals are close to the agent if goals are fixed."
        # '-> Moving the agent initial position closed to a fixed goal is not implemented.
        assert goal_range is None or 0 <= goal_range[0] <= goal_range[1]
        if "." not in behaviour_file:
            behaviour_file += ".acs"
        if "." not in configuration_file:
            configuration_file += ".cfg"
        assert behaviour_file.endswith(".acs"), "Wrong behaviour file extension, expected '.acs'"
        assert configuration_file.endswith(".cfg"), "Wrong behaviour file extension, expected '.cfg'"
        config_files_directory_path = os.path.dirname(os.path.abspath(__file__)) + "/configurations/"
        assert os.path.exists(config_files_directory_path + behaviour_file), \
            "Behaviour file not found. Verify that this file exists in " + config_files_directory_path + "."
        assert os.path.exists(config_files_directory_path + configuration_file), \
            "Behaviour file not found. Verify that this file exists in " + config_files_directory_path + "."
        assert isinstance(block_size, int) and block_size > 0
        assert isinstance(pixels_per_tile, int) and pixels_per_tile > 0
        assert isinstance(pixels_wall_width, int) and pixels_per_tile >= pixels_wall_width > 0
        assert isinstance(action_frame_repeat, int) and 0 < action_frame_repeat

        self.map_tag = map_tag
        self.map_name = map_tag.name.lower()
        self.reset_anywhere = reset_anywhere
        self.random_walls_textures = random_walls_textures
        self.sparse_reward = sparse_reward
        self.behaviour_file = behaviour_file
        self.configuration_file = configuration_file
        self.block_size = block_size
        self.attainability_threshold = attainability_threshold
        self.discrete_goals = discrete_goals
        self.goal_range = goal_range
        self.pixels_per_tile = pixels_per_tile
        self.pixels_wall_width = pixels_wall_width
        self.action_frame_repeat = action_frame_repeat
        self.observation_resolution = observation_resolution

        # Generate wad file and save it in a temp directory (prevent concurrent access if two
        # environment are created at the same time on the same wad file)
        self.wad_temporary_directory = TemporaryDirectory()
        self.map_array = np.array(
            importlib.import_module("environments.doom_visual_navigation.maps." + self.map_name).maze_array)
        print("Building wad file ... ", end="\r")
        build_wad(self.wad_temporary_directory.name, self.map_array,
                  random_walls_textures=self.random_walls_textures, behaviour_file=self.behaviour_file,
                  block_size=self.block_size, discrete_goals_set=self.discrete_goals)
        print("Building wad file ... DONE")

        # Build DOOM GAME !
        self.reset_game(with_wad=False)

        # Environment variables
        self.goal_tile_coordinates = None

        self.position_space = Box(low=np.zeros(2),
                                  high=np.array([self.map_array.shape[1] * self.block_size,
                                                 self.map_array.shape[0] * self.block_size]))
        available_buttons = self._game.get_available_buttons()
        self.action_space = Discrete(len(available_buttons))
        observation_shape = self.observation_resolution.name.split("_")[-1].split("X")
        observation_shape = tuple([3] + [int(elt) for elt in observation_shape])
        self.observation_space = Box(low=np.zeros(observation_shape), high=np.full(observation_shape, 255))

        # Goals
        self.goal_image = None
        if self.discrete_goals:
            self.goals_positions = list(zip(*np.where(self.map_array == TileType.TERMINAL.value)))

    @property
    def goal_env_position(self):
        return self.get_position_from_tile(*self.goal_tile_coordinates)

    def get_tile_from_position(self, x, y):
        """
        Compute the coordinates in self.map_array from the position in the viz doom maze.
        :param x: x coordinate in the viz doom maze.
        :param y: y coordinate in the viz doom maze.
        :return: tuple: Coordinates in the map_array.
        """
        h = (y - int(self.block_size / 2)) / self.block_size
        w = (x - int(self.block_size / 2)) / self.block_size
        h = len(self.map_array) - 1 - h
        return h, w

    def get_position_from_tile(self, h, w) -> (float, float):
        """
        Compute the position in the viz doom maze from the coordinates in self.map_array.
        :param w: x coordinate in the map_array.
        :param h: y coordinate in the map_array.
        :return: tuple: Position in the viz doom maze.
        """
        h = len(self.map_array) - 1 - h
        x, y = w * self.block_size, h * self.block_size
        x += int(self.block_size / 2)
        y += int(self.block_size / 2)
        return x, y

    def get_agent_position(self):
        """
        Return the agent position in the vizdoom maze.
        :return: tuple
        """
        observation = self._game.get_observation()
        agent_coordinates = np.array([observation.game_variables[0], observation.game_variables[1]])
        return agent_coordinates

    def distance_to_goal(self) -> float:
        """
        Compute the distance between the agent and the goal.
        :return: float
        """
        goal_coordinates = self.goal_env_position
        agent_coordinates = self.get_agent_position()
        return np.linalg.norm((goal_coordinates - agent_coordinates))

    def reset_game(self, with_wad=True):

        self._game = DoomGame()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._game.set_screen_resolution(self.observation_resolution)
        self._game.load_config(current_dir + "/configurations/" + self.configuration_file)

        if with_wad:
            # Reload the map
            self._game.set_doom_scenario_path(self.wad_temporary_directory.name + "/map.wad")
            self._game.set_doom_map("MAP00")
        self._game.init()

    def get_tiles_under_max_distance(self, h, w):
        """
        Find every tile in self.map_array that are at a distance of n from the agent initial tile.
        We will use a Dijkstra-like algorithm to find the distance between every tiles and the chosen start tile.

        :param h: Tile coordinate over the map width.
        :param w: Tile coordinate over the map height.
        :return: Tiles that are close enough from the given tile coordinates.
        """

        result = []
        seen = np.full_like(self.map_array, False)
        height, width = self.map_array.shape

        def __get_neighbors(h, w):
            neighbors = []
            potential_neighbors = [(h - 1, w), (h + 1, w), (h, w - 1), (h, w + 1)]
            for h_, w_ in potential_neighbors:
                if 0 <= h_ < height and 0 <= w_ < width and self.get_tile_type(h_, w_) != TileType.WALL:
                    neighbors.append((h_, w_))
            return neighbors

        def __explore(tile_h, tile_w, distance_so_far=0):
            """
            Recursive function that, according to a given tile, and a distance from a source tile, verify if this tile
            is not too far, and if so call itself recursively on the given tile neighbors.
            Because of the usage of a shared memory list "seen", a tile cannot be added twice in the result list.

            :param tile_h: Given tile height coordinate, aka. self.map_array[tile_h, tile_w] give the right tile.
            :param tile_w: Given tile width coordinate, aka. self.map_array[tile_h, tile_w] give the right tile.
            :param distance_so_far: Distance from the original call tile and the current one.
            :return: If called with distance_so_far=0, return a list of tiles coordinates that are at a distance
            between bounds defined by self.goal_range.
            """
            assert self.goal_range is not None and 0 <= self.goal_range[0] <= self.goal_range[1]
            if seen[tile_h, tile_w] or distance_so_far > self.goal_range[1]:
                return
            if distance_so_far >= self.goal_range[0]:
                result.append((tile_h, tile_w))
            seen[tile_h, tile_w] = True

            if distance_so_far + 1 > self.goal_range[1]:
                return  # No need to go further

            for neighbor in __get_neighbors(tile_h, tile_w):
                __explore(*neighbor, distance_so_far=distance_so_far + 1)

        __explore(h, w)
        return result

    def reset(self):

        # Load wad file
        wad = WAD(from_file=os.path.join(self.wad_temporary_directory.name, "map.wad"))
        map = wad.maps["MAP00"]
        map_editor = MapEditor(map)

        # Choose agent start tile.
        if self.reset_anywhere:
            agent_start_tile_candidates = np.where(self.map_array != 1)
        else:
            agent_start_tile_candidates = np.where(self.map_array == 2)
        agent_start_tile_candidates = list(zip(*agent_start_tile_candidates))
        agent_start_tile = random.choice(agent_start_tile_candidates)

        # Choose goal tile
        if self.discrete_goals:
            self.goal_tile_coordinates = random.choice(self.goals_positions)
            goal_x, goal_y = self.get_position_from_tile(*self.goal_tile_coordinates)
        else:
            if self.goal_range is not None:
                candidates = self.get_tiles_under_max_distance(*agent_start_tile)
            else:
                candidates = list(zip(*np.where(self.map_array != 1)))
            h, w = random.choice(candidates)
            assert self.get_tile_type(h, w) != TileType.WALL
            self.goal_tile_coordinates = (h, w)
            goal_x, goal_y = self.get_position_from_tile(h, w)

            # Find the things that belong to the goal
            for thing_id, thing in enumerate(map_editor.things):
                if thing.type == 32:
                    thing.x = goal_x
                    thing.y = goal_y
                    break

        # Get goal image
        # - Place the agent next to the goal facing it
        angle = random.random() * 2 * math.pi

        agent_x = goal_x + math.cos(angle) * self.attainability_threshold
        agent_y = goal_y + math.sin(angle) * self.attainability_threshold

        # Place agent
        for thing_id, thing in enumerate(map_editor.things):
            if thing.type == 1:
                thing.x = int(agent_x)
                thing.y = int(agent_y)
                thing.angle = int(math.degrees(angle + math.pi))
                break

        # Get agent pov
        self.reset_game()  # Reload the modified wad, so we can take a picture of the goal before to replace the agent
        self.goal_image = self.observation.copy()

        # - Get agent observation and store it as a goal to reach.

        # Place agent
        x, y = self.get_position_from_tile(*agent_start_tile)
        for thing_id, thing in enumerate(map_editor.things):
            if thing.type == 1:  # A thing with a type of 1 is the agent
                thing.x = x
                thing.y = y
                thing.angle = 0  # Arbitrary looking to the right, can be changed with a value between 0 and 360

        # Save the map,
        wad.maps['MAP00'] = map_editor.to_lumps()

        # Start the episode,
        # os.system('nautilus ' + str(self.wad_temporary_directory.name))
        self._game.new_episode()
        self.reset_game()  # Reload the modified wad, so we can take a picture of the goal before to replace the agent

        return self.observation, self.goal_image

    def step(self, action=None):

        if action is None:
            self._game.advance_action()
        else:
            actions = [0 for _ in range(self.action_space.n)]
            actions[action] = 1
            self._game.make_action(actions, self.action_frame_repeat)
        distance_to_goal = self.distance_to_goal()
        reached = distance_to_goal <= self.attainability_threshold

        if self.sparse_reward:
            reward = -1 if reached else 0
        else:
            reward = - distance_to_goal

        return self.observation, reward, reached, {"reached": reached}

    @property
    def observation(self):
        """
        Return the agent observation.
        """
        observation: Gameobservation = self._game.get_observation()
        observation = observation.screen_buffer.copy()
        return observation

    def get_tile_type(self, h, w):
        return TileType(self.map_array[h, w].item())

    def get_color(self, h, w, ignore_agent=False, ignore_terminals=False):
        agent_h, agent_w = self.get_tile_from_position(*self.get_agent_position())
        if (agent_h, agent_w) == (h, w) and not ignore_agent:
            return Colors.AGENT.value
        else:
            tile_type = self.get_tile_type(h, w)
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

    def reached(self):
        raise NotImplementedError()

    """
    Every functions bellow this are used for rendering.
    """

    def render(self, mode="top_view", ignore_goal=False, ignore_agent=False):
        assert mode in ["top_view", "observation", "observation_goal"]
        ignore_terminals = ignore_goal or not self.discrete_goals
        if mode == "top_view":
            height, width = self.map_array.shape

            # Compute the total grid size
            width_px = height * self.pixels_per_tile
            height_px = width * self.pixels_per_tile

            # Build an empty image
            img = np.full(fill_value=255, shape=(height_px, width_px, 3), dtype=np.uint8)

            # Render the grid
            for h in range(height):
                for w in range(width):
                    tile_type = self.get_tile_type(h, w)
                    position_1 = self.get_position_from_tile(h, w)

                    if tile_type == TileType.WALL:
                        neighbors = [(h + 1, w), (h, w + 1)]  # TODO supprimer les -1 et tester
                        for h_, w_ in neighbors:
                            if 0 <= h_ < height and 0 <= w_ < width and self.get_tile_type(h_, w_) == TileType.WALL:
                                position_2 = self.get_position_from_tile(h_, w_)
                                self.place_edge(img, *position_1, *position_2, Colors.WALL.value)

                        # If there is no walls next to this one, we still want to place a point (like a pillar)
                        # on this one.
                        self.place_point(img, *position_1, Colors.WALL.value, self.pixels_wall_width)
                    elif not ignore_terminals and tile_type == TileType.WALL:
                        img = self.place_point(img, *position_1, Colors.GOAL.value)
                    elif not ignore_goal and not self.discrete_goals and (h, w) == self.goal_tile_coordinates:
                        img = self.place_point(img, *position_1, Colors.GOAL.value)
            if not ignore_agent:
                img = self.place_point(img, *self.get_agent_position(), Colors.AGENT.value)
            return img

        elif mode == "observation":
            return np.moveaxis(self._game.get_observation().screen_buffer.copy(), 0, -1)
        elif mode == "observation_goal":
            observation = np.moveaxis(self._game.get_observation().screen_buffer.copy(), 0, -1)
            goal = np.moveaxis(self.goal_image.copy(), 0, -1)
            return np.concatenate((observation, goal), 1)

    def place_point(self, image: np.ndarray, position_x, position_y, color: Union[np.ndarray, list], width=5):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the observation space of the point to place.
        param y: y coordinate in the observation space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        position = np.array([position_x, position_y])
        if isinstance(color, list):
            color = np.array(color)

        observation_space_range = (self.position_space.high - self.position_space.low)
        center = (position - self.position_space.low) / observation_space_range
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

    def place_edge(self, image: np.ndarray, position_1_x, position_1_y, position_2_x, position_2_y,
                   color: Union[np.ndarray, list]):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the observation space of the point to place.
        param y: y coordinate in the observation space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """

        position_1 = np.array([position_1_x, position_1_y])
        position_2 = np.array([position_2_x, position_2_y])

        color = np.array(color) if isinstance(color, list) else color
        observation_space_range = (self.position_space.high - self.position_space.low)

        center = (position_1 - self.position_space.low) / observation_space_range
        center[1] = 1 - center[1]
        center_y_1, center_x_1 = (image.shape[:2] * np.flip(center)).astype(int)

        center = (position_2 - self.position_space.low) / observation_space_range
        center[1] = 1 - center[1]
        center_y_2, center_x_2 = (image.shape[:2] * np.flip(center)).astype(int)

        rr, cc, val = line_aa(center_y_1, center_x_1, center_y_2, center_x_2)
        old = image[rr, cc]
        extended_val = np.tile(val, (3, 1)).T
        image[rr, cc] = (1 - extended_val) * old + extended_val * color

        return image


if __name__ == "__main__":
    # TODO
    # - Sampler des buts à une distance max
    test = True
    environment = DoomMaze(map_tag=DoomMapIndex.IMPOSSIBLE,
                           reset_anywhere=False,
                           discrete_goals=False,
                           goal_range=(15, 20),
                           test=test,
                           pixels_per_tile=5)

    results = []
    output_directory = os.path.dirname(os.path.abspath(__file__))
    for i in range(10):
        print("Episode #" + str(i + 1))
        images = []

        environment.reset()

        goal_image = environment.goal_image.copy()
        save_image(goal_image, output_directory + "/goals", "goal_episode_" + str(i))

        results.append(0)
        done = False
        for interaction_id in range(250):
            action = None if test else environment.action_space.sample()
            observation, reward, done = environment.step(action)

            # images.append(observation)
            image = environment.render(mode="top_view")
            images.append(image)
            if done:
                results[-1] = 1
                break
        print("results = ", results)
        generate_video(images, output_directory, "episode_" + str(i))

        print("Episode finished!")
        print("************************")
    a = 1
