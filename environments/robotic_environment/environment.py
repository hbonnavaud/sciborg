import time
from enum import Enum
import random
from multiprocessing import Process
import matplotlib
from matplotlib import pyplot as plt
import math

from skimage.draw import line_aa

from environments.robotic_environment.launch_gazebo import launch_gazebo
import numpy as np
from gym.spaces import Box
from std_srvs.srv import Empty
from environments.robotic_environment.simulations_assets.build_world_from_map import (generate_xml,
                                                                                      simulation_pos_to_maze_pos,
                                                                                      maze_pos_to_simulation_pos)
from environments.goal_conditioned_environment import GoalConditionedEnvironment
import rclpy
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile
from utils import get_euler_from_quaternion, place_line


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    REWARD = 3


class RobotsIndex(Enum):
    IROBOT = "irobot"


class RobotMapsIndex(Enum):
    EMPTY = "empty_room"
    FOUR_ROOMS = "four_rooms"
    HARD = "hard_maze"
    MEDIUM = "medium_maze"
    JOIN_ROOMS = "join_rooms_medium"


class SingleRobotEnvironment(GoalConditionedEnvironment):

    def __init__(self,
                 # simulation parameters
                 map_name: str = RobotMapsIndex.FOUR_ROOMS.value,
                 robot_name: str = RobotsIndex.IROBOT.value,
                 environment_size_scale: float = 0.5,
                 headless: bool = False,
                 nb_markers: int = 0,

                 # Environment behaviour settings
                 real_time: bool = True,
                 simulation_step_duration: float = 0.1,  # UNUSED if real_time=True
                 goal_reachability_threshold: float = 0.40,
                 collision_distance_threshold: float = 0.30,
                 reward_at_collision: float = None,  # Set the reward given at collision.
                 reward_once_reached: float = 0,
                 sparse_reward: bool = True,
                 max_velocity: np.ndarray = np.array([0.5, 1]),
                 max_action: np.ndarray = np.array([0.2, 0.4]),

                 # State composition and settings
                 use_lidar: bool = False,
                 lidar_max_angle: float = None,  # USELESS if variable use_lidar is false
                 nb_lidar_beams: int = 20,  # USELESS if variable use_lidar is false
                 use_odometry: bool = False  # Use odometry topic as position/orientation (instead of /pose by default)
                 ):

        """
        # Simulation settings
        :params map_name: str, name of the map, the environment will load the map located at ./simulation_assets/maps/
            {map_name}.py
        :params robot_name: str, name of the robot, the environment will load the map located at ./simulation_assets/
            maps/worlds/building_assets/robots/{robot_name}.xml
        :params environment_size_scale: float, size of an element of the map_array, in the gazebo .world file.
        :params headless: bool, whether to launch the gazebo client or not
        :params nb_markers: int, default=0, how many marker should be added in the .world file. Markers are static
            sphere models without collision, used to mark a position like a goal or an RRT graph node for example.

        # Environment behaviour settings
        :params real_time: bool, whether to pause the simulation, and play it for a fixed duration at each step, or let
            it run and interact with it in real time.
        :params simulation_step_duration: float, if real_time is True, this variable define the duration of a
            simulation step in second (cf. real_time description)
        :params goal_reachability_threshold: float, distance under which the goal will be considered as reached.

        :params collision_distance_threshold: float, distance under which an obstacle will be considered as hit.
        :params reward_at_collision: float, reward to give to the agent when a collision is detected.
            If None, no reward is the same that for a default step.
        :params sparse_reward: bool, if true, the agent will receive a reward of -1 at each step. It will receive a
            reward of - euclidean distance otherwise.
        :params max_velocity: np.ndarray, Maximum velocity that the agent can have. The lower bound is - max_velocity.
        :params max_action: np.ndarray, Maximum action (aka. velocity evolution) to put on the agent at each step,
            which can be seen as the maximum acceleration. The lower bound is - max_action.

        # State composition and settings
        :params use_lidar: bool, whether to add the lidar information in the agent's observation or not.
        :params lidar_max_angle: float, UNUSED IF use_lidar=false. Maximum angle of the lidar x. Aka. the lidar will
            scan an area from -x to x (in radiant). lidar_max_angle=None (default) will set this value to the maximum
            angle possible. On irobot, it is around 2.35, aka. 0.74 * pi, which leads to a 266.4° scan.
        :params nb_lidar_beams: int, UNUSED IF use_lidar=false. the number of beams that compose the lidar information
            given to the agent. If the number of beams received on the ros topic n is bigger than nb_lidar_beams m,
            the ith value sent to the agent will be the minimal value of the received beams from the (n/m * i)th to the
            (n/m * (i + 1))th.
        :params use_odometry: bool, if true, use odometry topic as position/orientation. Use /pose as position and
            orientation otherwise.
        """
        assert goal_reachability_threshold > collision_distance_threshold, \
            "How do you want to reach a goal that is next to a wall?"

        self.environment_size_scale = environment_size_scale
        self.nb_markers = nb_markers
        self.real_time = real_time
        self.simulation_step_duration = simulation_step_duration
        self.goal_reachability_threshold = goal_reachability_threshold
        self.collision_distance_threshold = collision_distance_threshold
        self.reward_at_collision = reward_at_collision
        self.reward_once_reached = reward_once_reached
        self.sparse_reward = sparse_reward
        self.lidar_max_angle = lidar_max_angle
        self.nb_lidar_beams = nb_lidar_beams
        self.use_lidar = use_lidar
        self.use_odometry = use_odometry

        rclpy.init()
        self.node = rclpy.create_node('robotic_environment')

        # Build .world file from the given map
        self.maze_array, world_file_path = generate_xml(map_name, robot_name, scale=self.environment_size_scale,
                                                        nb_markers=self.nb_markers)

        # Launch gazebo using the launch file and the robot
        p = Process(target=launch_gazebo, args=(str(world_file_path), headless,))
        # TODO add robot sdf file, maybe later
        #  NB1: gave it a try, took me four hours without success. Good luck.
        #  There might be a trick I don't know though.
        p.start()

        # At this time, gazebo is launching. We wait until we can find the mandatory reset_world service (which means
        # that gazebo is fully launched). Looking into 'ps aux' result don't work since the topics are created after the
        # gazebo process.
        time.sleep(3)  # Gazebo launching will take some time. Wait a bit before to start looking for topic.
        reset_service_found = False
        i = 0
        while not reset_service_found:
            print("Looking for gazebo ..." + str(i))
            i += 1
            for service in self.node.get_service_names_and_types():
                if service[0].split("/")[-1] == "reset_world":
                    reset_service_found = True
                    assert service[1][0] == "std_srvs/srv/Empty", "Wrong type for reset service."
                    break
            time.sleep(0.5)

        # Look for topics
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST,
                                depth=1)
        lidar_message_type = "sensor_msgs/msg/LaserScan"
        position_message_type = "nav_msgs/msg/Odometry" if self.use_odometry else "geometry_msgs/msg/PoseStamped"
        commands_message_type = "geometry_msgs/msg/Twist"
        self.last_lidar_message = None
        self.last_position_message = None
        self.last_observation = None
        for topic in self.node.get_topic_names_and_types():
            topic_name = topic[0]
            topic_type = topic[1][0]
            if lidar_message_type == topic_type:
                self.lidar_subscriber = self.node.create_subscription(LaserScan, topic_name, self._lidar_callback,
                                                                      qos_profile=qos_sensor)
            elif position_message_type == topic_type:
                self.position_message_type = Odometry if self.use_odometry else PoseStamped
                self.position_subscriber = self.node.create_subscription(self.position_message_type, topic_name,
                                                                         self._position_callback,
                                                                         qos_profile=qos_sensor)
            elif commands_message_type == topic_type:
                self.commands_publisher = self.node.create_publisher(Twist, topic_name, QoSProfile(depth=10))

        # Find missing topics
        if not hasattr(self, "lidar_subscriber"):
            raise EnvironmentError("No topic found with type '" + lidar_message_type + "' alone for lidar.")
        if not hasattr(self, "position_subscriber"):  # MANDATORY
            raise EnvironmentError("No topic found with type '" + position_message_type + "' alone for position.")
        if not hasattr(self, "commands_publisher"):  # MANDATORY
            raise EnvironmentError("No topic found with type '" + commands_message_type + "' alone for commands.")

        # Find required service names
        for service in self.node.get_service_names_and_types():
            service_name = service[0]
            service_type = service[1][0]
            if service_type == "std_srvs/srv/Empty":
                if service_name.split("/")[-1].startswith("pause"):
                    self.pause_client = self.node.create_client(Empty, service_name)
                elif service_name.split("/")[-1].startswith("unpause"):
                    self.unpause_client = self.node.create_client(Empty, service_name)
                if service_name.split("/")[-1] == "reset_world":
                    self.reset_client = self.node.create_client(Empty, service_name)
            elif service_name.endswith("set_entity_state"):
                self.set_entity_state_client = self.node.create_client(SetEntityState, service_name)

        if not hasattr(self, "pause_client"):
            raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service pause.")
        if not hasattr(self, "unpause_client"):
            raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service unpause.")
        if not hasattr(self, "reset_client"):
            raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service reset_world.")

        # Setup spaces

        """
        Observation:
        o[0:3]: x, y, z, position
        o[3:7]: quaternion orientation
        o[7]: linear velocity
        o[8]: angular velocity
        o[9:]: lidar beams for the next n values if use_lidar is true, where n is self.nb_lidar_beams.
        """

        self.observation_size = 9
        if self.use_lidar:
            self.observation_size += self.nb_lidar_beams

        xy_high = np.full(2, - 0.5)
        xy_high = np.array(maze_pos_to_simulation_pos(*xy_high, self.environment_size_scale, self.maze_array))
        others_high = np.full(self.observation_size - 2, float("inf"), dtype=np.float16)
        high = np.concatenate((xy_high, others_high))
        self.observation_space = Box(-high, high, dtype=np.float16)
        self.goal_space = Box(-xy_high, xy_high, dtype=np.float16)
        # \-> WARNING Sample in the goal space do not guarantee that the sampled goal will not be inside a wall.
        #             Check the method in self.reset function.
        self.goal = np.zeros(self.goal_space.shape)

        self.waiting_for_position = False
        self.waiting_for_lidar = False

        # Velocities
        self.max_velocity = max_velocity
        self.action_space = Box(low=- max_action, high=max_action, dtype=np.float32)
        self.velocity = np.zeros(2)  # self.velocity = np.array([linear_velocity, angular_velocity])

        if self.last_position_message is None:
            print("Waiting for position topic ...")
            while self.last_position_message is None:
                rclpy.spin_once(self.node)
                time.sleep(0.1)

        if self.last_lidar_message is None:
            # \-> Even if self.use_lidar is false, because we will use lidar to detect collisions
            print("Waiting for lidar topic ...")
            while self.last_lidar_message is None:
                rclpy.spin_once(self.node)
                time.sleep(0.1)

    def get_observation(self):

        #### GET STATE ####
        self.waiting_for_position = True
        self.waiting_for_lidar = True
        while self.waiting_for_position and self.waiting_for_lidar:
            rclpy.spin_once(self.node)

        agent_observation = np.array([
            self.last_position_message.pose.pose.position.x,
            self.last_position_message.pose.pose.position.y,
            self.last_position_message.pose.pose.position.z,
            self.last_position_message.pose.pose.orientation.w,
            self.last_position_message.pose.pose.orientation.x,
            self.last_position_message.pose.pose.orientation.y,
            self.last_position_message.pose.pose.orientation.z,
            self.velocity[0].item(),
            self.velocity[1].item()
        ])

        # read laser state
        if self.use_lidar:
            assert self.last_lidar_message
            if self.lidar_max_angle is not None:
                # Filter lidar ranges
                angle = self.last_lidar_message.angle_min
                ranges = []
                for r in self.last_lidar_message.ranges:
                    if - self.lidar_max_angle > angle < self.lidar_max_angle:
                        continue
                    ranges.append(r)
                    angle += self.last_lidar_message.angle_increment
            else:
                # Keep the original ones
                ranges = self.last_lidar_message.ranges

            # Sample self.nb_lidar_beams beams
            beam_width = len(ranges) // self.nb_lidar_beams
            beams = []
            for i in range(self.nb_lidar_beams):
                beams.append(ranges[beam_width * i + beam_width // 2])
            agent_observation = np.concatenate((agent_observation, np.array(beams)))

        # Detect collision
        collided = np.array(self.last_lidar_message.ranges).min() < self.collision_distance_threshold
        return agent_observation, collided

    def step(self, action):

        #### PERFORM ACTION ####

        # Publish the robot action
        if action is not None:  # Handy if you want to control the robot with a telehop script in parallel
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.velocity = np.clip(self.velocity + action, - self.max_velocity, self.max_velocity)
            vel_cmd = Twist()
            vel_cmd.linear.x = self.velocity[0].item()
            vel_cmd.angular.z = self.velocity[1].item()
            self.commands_publisher.publish(vel_cmd)

        # Unpause the simulation
        if not self.real_time:
            self.unpause_client.call_async(Empty.Request())
            time.sleep(self.simulation_step_duration)
            self.pause_client.call_async(Empty.Request())

        observation, collided = self.get_observation()

        # Compute distance to the goal, and compute reward from it
        distance = np.linalg.norm(observation[:2] - self.goal)
        reached = distance < self.goal_reachability_threshold
        malus = -1 if self.sparse_reward else - distance
        reward = 0 if reached else malus
        self.last_observation = observation
        return observation, reward, reached, {"collided": collided, "reached": reached}

    def reset(self):
        self.reset_client.call_async(Empty.Request())

        # Sample a goal position from reachable tiles
        goal_tile = random.choice(np.argwhere(self.maze_array != TileType.WALL.value))
        goal_position = maze_pos_to_simulation_pos(*goal_tile, self.environment_size_scale, self.maze_array)
        goal_position = np.array(goal_position)
        self.goal = np.random.uniform(goal_position - self.environment_size_scale / 2,
                                      goal_position + self.environment_size_scale / 2)
        # self.environment_size_scale / 2 aka: half the size of a maze_array tile.
        # In other words, we sample a goal uniformly inside the tile we sampled.

        # Move the goal marker to the goal position, so we can see the goal in the gazebo simulation:
        request = SetEntityState.Request()
        request.state.name = "goal_marker"
        request.state.pose = Pose(position=Point(x=self.goal[0], y=self.goal[1], z=0.2))
        self.set_entity_state_client.call_async(request)

        observation, collided = self.get_observation()
        assert not collided, ("Agent spawned on a wall, there should be an issue somewhere. Verify collision detection "
                              "function, and verify where the agent spawn in the env (enable client mode with "
                              "headless=False)")

        return observation, self.goal

    ##########################################################################
    ####                       SUBSCRIBERS CALLBACKS                      ####
    ##########################################################################
    def _lidar_callback(self, msg):
        # Verify collision:
        mini = np.min(msg.ranges)
        if mini < self.collision_distance_threshold:
            self.commands_publisher.publish(Twist())

        self.last_lidar_message = msg
        self.waiting_for_lidar = False

    def _position_callback(self, msg):
        self.last_position_message = msg
        self.waiting_for_position = False

    def get_goal_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[:2]

    def pause_simulation(self):
        self.pause_client.call_async(Empty.Request())

    def unpause_simulation(self):
        self.unpause_client.call_async(Empty.Request())

    def render(self, resolution=10, ignore_goal=False) -> np.ndarray:
        """
        :params pixels_tiles_width: int, tiles width in pixel.
        :params ignore_goal: bool, whether to draw goal on the image or not.
        Return a np.ndarray of size (width, height, 3) of pixels that represent the environments and it's walls
        :return: The final image.
        """

        # we add np.full(2, 2) to get image borders, because border walls are not included in env.maze_space.
        # We assume walls have a width of one, but it's not really important for a generated image.

        self.pause_simulation()

        image_width_px, image_height_px = np.array(self.maze_array.shape) * resolution
        image = np.full((image_height_px, image_width_px, 3), 255, dtype=np.uint8)

        # Render the grid
        for coordinates in np.argwhere(self.maze_array == TileType.WALL.value):
            image[coordinates[0] * resolution:(coordinates[0] + 1) * resolution,
                  coordinates[1] * resolution:(coordinates[1] + 1) * resolution, :] = TileType.WALL.value

        def place_point(x, y, color=None, width=resolution / 2):
            nonlocal image, resolution
            if isinstance(width, float):
                width = int(width)
            if color is None:
                color = [125, 125, 125]
            if isinstance(color, list):
                color = np.array(color)

            center = simulation_pos_to_maze_pos(x, y, self.environment_size_scale, self.maze_array) + np.full(2, 0.5)
            center = (np.array(center) * resolution).astype(int)
            assert (center >= 0).all()

            # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
            # each pixel inside this square to
            radius = width
            for i in range(center[0] - radius, center[0] + radius):
                for j in range(center[1] - radius, center[1] + radius):
                    dist = np.linalg.norm(np.array([i, j]) - center)
                    if dist < radius and 0 <= i < image.shape[1] and 0 <= j < image.shape[0]:
                        image[i, j] = color
            return image

        if not ignore_goal:
            place_point(*self.goal, color=[255, 0, 0], width=resolution / 2)

        # Place orientation
        agent_color = [0, 0, 255]
        orientation = get_euler_from_quaternion(*self.last_observation[3:7])
        position_1 = np.array(simulation_pos_to_maze_pos(*self.last_observation[:2], self.environment_size_scale,
                                                         self.maze_array)) + np.full(2, 0.5)
        position_2 = position_1 + np.array([math.cos(orientation[0]), - math.sin(orientation[0])])
        start_center_pixel_x, start_center_pixel_y = (position_1 * resolution).astype(int)
        stop_center_pixel_x, stop_center_pixel_y = (position_2 * resolution).astype(int)
        rr, cc, val = line_aa(start_center_pixel_x, start_center_pixel_y, stop_center_pixel_x, stop_center_pixel_y)
        old = image[rr, cc]
        extended_val = np.tile(val, (3, 1)).T
        image[rr, cc] = (1 - extended_val) * old + extended_val * agent_color

        if self.last_observation is None:
            self.step(np.array([0, 0]))
        place_point(*self.last_observation[:2], color=agent_color, width=resolution / 2)

        self.unpause_simulation()

        return image


if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    env = SingleRobotEnvironment(map_name=RobotMapsIndex.EMPTY.value, use_lidar=True, use_odometry=True,
                                 real_time=False)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    env.reset()
    env.pause_simulation()
    print("waiting")
    env.unpause_simulation()
    env.reset()

    for step_id in range(5000):
        print("Step {}".format(step_id))
        agent_observation, reward_, done_, info_ = env.step(None)
        collided_ = info_["collided"]
        reached_ = info_["reached"]
        img = env.render()
        ax.cla()
        ax.imshow(img)
        fig.canvas.flush_events()
        if reached_ or done_:
            print("\n\n GOAL REACHED \n\n")
            break



