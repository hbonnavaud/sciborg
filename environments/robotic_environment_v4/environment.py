import subprocess
import time
from enum import Enum
import random
from multiprocessing import Process
from typing import Union

import matplotlib
from matplotlib import pyplot as plt
import math

from skimage.draw import line_aa

from .launch_gazebo import launch_gazebo
import numpy as np
from gym.spaces import Box
from std_srvs.srv import Empty
from .simulations_assets.build_world_from_map import generate_xml
from sciborg.environments.goal_conditioned_environment import GoalConditionedEnvironment
import rclpy
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile
from sciborg.utils import quaternion_to_euler


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
    VOLIERE = "voliere"


class SingleRobotEnvironmentV4(GoalConditionedEnvironment):

    def __init__(self,
                 # simulation parameters
                 map_name: str = RobotMapsIndex.FOUR_ROOMS.value,
                 robot_name: str = RobotsIndex.IROBOT.value,
                 environment_size_scale: float = 0.5,
                 headless: bool = False,
                 nb_markers: int = 0,

                 # Environment behaviour settings
                 real_time: bool = True,
                 simulation_step_duration: float = 0.15,  # UNUSED if real_time=True
                 goal_reachability_threshold: float = 0.40,
                 collision_distance_threshold: float = 0.3,
                 lidar_to_center_distance: float = 0.1,
                 reward_at_collision: float = None,  # Set the reward given at collision.
                 reward_once_reached: float = 0,
                 sparse_reward: bool = True,
                 max_velocity: np.ndarray = np.array([0.5, 1]),
                 max_action: np.ndarray = np.array([0.2, 0.4]),

                 # State composition and settings
                 use_lidar: bool = False,
                 lidar_max_angle: float = None,  # USELESS if variable use_lidar is false
                 nb_lidar_beams: int = 20,  # USELESS if variable use_lidar is false

                 real_world=False  # Whether the environment is used in real-world or not.
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
        self.lidar_to_center_distance = lidar_to_center_distance
        self.reward_at_collision = reward_at_collision
        self.reward_once_reached = reward_once_reached
        self.sparse_reward = sparse_reward
        self.lidar_max_angle = lidar_max_angle
        self.nb_lidar_beams = nb_lidar_beams
        self.use_lidar = use_lidar
        self.real_world = real_world

        if rclpy.ok():
            rclpy.shutdown()
        rclpy.init()
        self.node = rclpy.create_node('robotic_environmentV2')

        # Build .world file from the given map
        self.maze_info, world_file_path = generate_xml(map_name, robot_name, scale=self.environment_size_scale,
                                                       nb_markers=self.nb_markers)

        if self.real_world:
            self.available_goals = [np.array([0., 0.])]
        else:

            # Launch gazebo server using the launch file and the robot
            self.gazebo_subprocess = Process(target=launch_gazebo, args=(str(world_file_path),))
            self.gazebo_subprocess.start()
            # TODO add robot sdf file, maybe later
            #  NB1: gave it a try, took me four hours without success. Good luck.
            #  There might be a trick I don't know though.

            # At this time, gazebo is launching. We wait until we can find the mandatory reset_world service 
            # (whic!h means that gazebo is fully launched). Looking into 'ps aux' result don't work since the topics
            # are created after the gazebo process.
            time.sleep(3)  # Gazebo launching will take some time. Wait a bit before to start looking for topic.
            reset_service_found = False
            while not reset_service_found:
                print("Waiting for gazebo ...", end="\r")
                for service in self.node.get_service_names_and_types():
                    if service[0].split("/")[-1] == "reset_world":
                        reset_service_found = True
                        assert service[1][0] == "std_srvs/srv/Empty", "Wrong type for reset service."
                        break
            time.sleep(0.5)

            print("Waiting for gazebo ... DONE")

        # Look for topics
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST,
                                depth=1)

        self.last_lidar_message = None
        self.last_position_message = None
        self.last_observation = None
        self.commands_publisher = None

        if self.real_world:
            command_topic = {"name": "/cmd_vel", "type": "geometry_msgs/msg/Twist"}
            lidar_topic = {"name": "/scan", "type": "sensor_msgs/msg/LaserScan"}
            position_topic = {"name": "/vrpn_mocap/IRobot_02/pose", "type": "geometry_msgs/msg/PoseStamped"}

            print("Looking for topics ...")
            time.sleep(2)  # Let some time to discovery to happen (issue https://github.com/ros2/ros2/issues/1057)
            topics = self.node.get_topic_names_and_types()
            for topic in topics:
                topic_name = topic[0]
                topic_type = topic[1][0]

                if topic_name == command_topic["name"] and topic_type == command_topic["type"]:
                    print("Found command topic")
                    self.commands_publisher = self.node.create_publisher(Twist, topic_name, QoSProfile(depth=10))
                if topic_name == lidar_topic["name"] and topic_type == lidar_topic["type"]:
                    print("Found lidar topic")
                    self.lidar_subscriber = self.node.create_subscription(LaserScan, topic_name, self._lidar_callback,
                                                                          qos_profile=qos_sensor)
                if topic_name == position_topic["name"] and topic_type == position_topic["type"]:
                    print("Found position topic")
                    self.position_message_type = PoseStamped if self.real_world else Odometry
                    self.position_subscriber = self.node.create_subscription(self.position_message_type, topic_name,
                                                                             self._position_callback,
                                                                             qos_profile=qos_sensor)
        else:
            lidar_message_type = "sensor_msgs/msg/LaserScan"
            position_message_type = "nav_msgs/msg/Odometry"
            commands_message_type = "geometry_msgs/msg/Twist"

            detected_topics = self.node.get_topic_names_and_types()
            print("Looking for topics ...")
            for topic in self.node.get_topic_names_and_types():
                topic_name = topic[0]
                topic_type = topic[1][0]
                if lidar_message_type == topic_type:
                    print("Found lidar topic")
                    self.lidar_subscriber = self.node.create_subscription(LaserScan, topic_name, self._lidar_callback,
                                                                          qos_profile=qos_sensor)
                elif position_message_type == topic_type:
                    print("Found position topic")
                    self.position_message_type = PoseStamped if self.real_world else Odometry
                    self.position_subscriber = self.node.create_subscription(self.position_message_type, topic_name,
                                                                             self._position_callback,
                                                                             qos_profile=qos_sensor)
                elif commands_message_type == topic_type:
                    print("Found command topic")
                    self.commands_publisher = self.node.create_publisher(Twist, topic_name, QoSProfile(depth=10))

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
        o[3:7]: x, y, z, w, quaternion orientation
        o[7]: linear velocity
        o[8]: angular velocity
        o[9:]: lidar beams for the next n values if use_lidar is true, where n is self.nb_lidar_beams.
        """

        self.observation_size = 9
        if self.use_lidar:
            self.observation_size += self.nb_lidar_beams

        if self.maze_info is None:
            high = np.full(self.observation_size, float("inf"), dtype=np.float32)
            self.observation_space = Box(-high, high)
            self.goal_space = Box(-high[:2], high[:2])
        else:
            xy_low = np.array([self.maze_info["x_min"], self.maze_info["y_min"]]).astype(np.float32)
            xy_high = np.array([self.maze_info["x_max"], self.maze_info["y_max"]]).astype(np.float32)

            inf = np.full(self.observation_size - 2, float("inf"), dtype=np.float32)
            high, low = np.concatenate((xy_high, inf)), np.concatenate((xy_low, -inf))
            self.observation_space = Box(low, high)
            self.goal_space = Box(xy_low, xy_high)
            # \-> WARNING Sample in the goal space do not guarantee that the sampled goal will not be inside a wall.
            #             Check the method in self.reset function.
        self.goal = np.zeros(self.goal_space.shape)

        self.waiting_for_position = False
        self.waiting_for_lidar = False

        # Velocities
        self.max_velocity = max_velocity
        max_action = max_action.astype(np.float32)
        self.action_space = Box(low=-max_action, high=max_action, dtype=np.float32)
        self.velocity = np.zeros(2)  # self.velocity = np.array([linear_velocity, angular_velocity])

        detected_topics = self.node.get_topic_names_and_types()  # For debugging in case of asserts bellow
        if self.last_position_message is None:
            print("Waiting for position topic ...", end="\r")
            while self.last_position_message is None:
                rclpy.spin_once(self.node)
                time.sleep(0.1)
            print("Waiting for position topic ... DONE")

        if self.last_lidar_message is None:
            assert "lidar_subscriber" in vars(self).keys()
            # \-> Even if self.use_lidar is false, because we will use lidar to detect collisions
            print("Waiting for lidar topic ...", end="\r")
            while self.last_lidar_message is None:
                rclpy.spin_once(self.node)
                time.sleep(0.1)
            print("Waiting for lidar topic ... DONE")
        if not self.real_world:
            self.hide_marker()
        print("Environment initialised.")

    def get_observation(self, compute_collision=True):

        #### GET STATE ####
        self.waiting_for_position = True
        self.waiting_for_lidar = True
        nb_trials = 0
        while self.waiting_for_position and self.waiting_for_lidar:
            rclpy.spin_once(self.node)
            nb_trials += 1
            if nb_trials > 20:
                print("did " + str(nb_trials) + " trials to get observation.", end="\r")

        if nb_trials > 20:
            print("got observation.")

        if self.real_world:
            agent_position = np.array([
                self.last_position_message.pose.position.x,
                self.last_position_message.pose.position.y,
                self.last_position_message.pose.position.z,
                self.last_position_message.pose.orientation.x,
                self.last_position_message.pose.orientation.y,
                self.last_position_message.pose.orientation.z,
                self.last_position_message.pose.orientation.w
            ])
        else:
            agent_position = np.array([
                self.last_position_message.pose.pose.position.x,
                self.last_position_message.pose.pose.position.y,
                self.last_position_message.pose.pose.position.z,
                self.last_position_message.pose.pose.orientation.x,
                self.last_position_message.pose.pose.orientation.y,
                self.last_position_message.pose.pose.orientation.z,
                self.last_position_message.pose.pose.orientation.w
            ])

        obs = np.concatenate((
            agent_position,
            np.array([
                self.velocity[0].item(),
                self.velocity[1].item()
            ])
        ))

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
            obs = np.concatenate((obs, np.array(beams)))

        if compute_collision:
            collided = False
            # self.velocity = - np.ones(2)
            # Detect collision
            collision_distance_th = self.collision_distance_threshold
            if self.velocity[0] < 0:
                collision_distance_th += 0.08
            orientation = quaternion_to_euler(obs[3:7])[-1].item()
            if self.velocity[0] != 0:
                nb_future_points = 10
                future_points = []
                last_point = obs[:2]
                new_point_offset = (np.array([np.cos(orientation), np.sin(orientation)])
                                    * (collision_distance_th / nb_future_points))
                for i in range(nb_future_points):
                    if self.velocity[0] > 0:
                        last_point = last_point + new_point_offset
                    else:
                        last_point = last_point - new_point_offset
                    future_points.append(last_point)

                for obstacle in self.maze_info["obstacles"]:
                    if obstacle["type"] == "rectangle":
                        obstacle_x = float(obstacle["position"].split(" ")[0])
                        obstacle_y = float(obstacle["position"].split(" ")[1])
                        size_x = float(obstacle["size"].split(" ")[0])
                        size_y = float(obstacle["size"].split(" ")[1])
                        box = Box(low=np.array([obstacle_x - size_x / 2, obstacle_y - size_y / 2]).astype(np.float32),
                                  high=np.array([obstacle_x + size_x / 2, obstacle_y + size_y / 2]).astype(np.float32))
                        for fp in future_points:
                            collided = box.contains(fp.astype(box.dtype))
                            if collided:
                                break
                        if collided:
                            break
                    else:
                        raise NotImplementedError()

                if not collided:
                    # Verify we're not going outside the environment
                    for fp in future_points:
                        if (fp[0] < self.maze_info["x_min"] or self.maze_info["x_max"] < fp[0]
                                or fp[1] < self.maze_info["y_min"] or self.maze_info["y_max"] < fp[1]):
                            collided = True
                            break

            if not collided:
                # fig, ax = plt.subplots()
                # Verify if the lidar detect something
                ranges_to_care_of = []
                debug_ranges = []
                angle = self.last_lidar_message.angle_min
                for r in self.last_lidar_message.ranges:
                    # There is a gap between lidar position on the robot and the center of the robot.
                    # The range of each laser beam should be adapter to simulate that the lidar is on the center of
                    # the robot.
                    # EXPLANATION: We use the law of cosines to compute the distance between the obstacle and the
                    # center, from:
                    #  - the distance between the lidar and the center,
                    #  - the distance between the lidar and the obstacle,
                    #  - the angle to the obstacle
                    new_range = math.sqrt(self.lidar_to_center_distance ** 2 + r ** 2
                                          - 2 * self.lidar_to_center_distance * r * math.cos(math.pi - angle))
                    if (r > 0.05 and  # r > 0.1 in order to remove errors
                            (self.velocity[0] < 0 and (- math.pi / 2 - 0.05 >= angle or angle >= math.pi / 2 + 0.05)
                             or self.velocity[0] > 0 and (- math.pi / 2 - 0.05 <= angle <= math.pi / 2 + 0.05)
                             or self.velocity[0] == 0)):
                        ranges_to_care_of.append(new_range)
                    # if r > 0.05 or True:
                    #     debug_ranges.append({"range": r, "new_range": new_range,
                    #                          "angle": angle, "position": len(ranges_to_care_of)})
                    angle += self.last_lidar_message.angle_increment

                # # RANGE
                # debug_ranges_np = np.array([elt["range"] for elt in debug_ranges])
                # argmin = np.argmin(debug_ranges_np)
                # mini = np.min(debug_ranges_np)
                # angle = debug_ranges[argmin]["angle"]
                # # NEW RANGE
                # new_debug_ranges_np = np.array([elt["new_range"] for elt in debug_ranges])
                # new_argmin = np.argmin(new_debug_ranges_np)
                # new_mini = np.min(new_debug_ranges_np)
                # new_angle = debug_ranges[new_argmin]["angle"]

                # NEW RANGE
                ranges = np.array(ranges_to_care_of)
                # if ranges_to_care_of:
                #     cared_argmin = np.argmin(ranges)
                #     cared_mini = np.min(ranges)
                if ranges_to_care_of and np.min(ranges) < collision_distance_th:
                    collided = True
                    if np.min(ranges) < 0.2:
                        print("Weired range")
        return (obs, collided) if compute_collision else obs

    def step(self, action):
        """
        Execute one step of the environment.
        Args:
            action: np.ndarray of shape (2,), the action to send to the robot

        Returns:
            observation, reward, done, info
        """

        #### PERFORM ACTION ####
        # Publish the robot action
        if action is not None:  # Handy if you want to control the robot with a telehop script in parallel
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.velocity = np.clip(self.velocity + action, - self.max_velocity, self.max_velocity)

            # Verify velocity: Put velocity to 0 if the robot id going out of bounds or is going in an obstacle
            observation, collided = self.get_observation()
            if collided:
                self.velocity = np.array([-0.5 if self.velocity[0] > 0 else 0.5, 1])

            vel_cmd = Twist()
            vel_cmd.linear.x = self.velocity[0].item()
            vel_cmd.angular.z = self.velocity[1].item()

            self.commands_publisher.publish(vel_cmd)

            if collided:
                time.sleep(1)

        # Unpause the simulation
        if not self.real_time:
            self.unpause_simulation()
        time.sleep(self.simulation_step_duration)
        if not self.real_time:
            self.pause_simulation()

        observation = self.get_observation(compute_collision=False)  # Ignore collided

        # Compute distance to the goal, and compute reward from it
        distance = np.linalg.norm(observation[:2] - self.goal)
        reached = distance < self.goal_reachability_threshold
        malus = -1 if self.sparse_reward else - distance
        reward = 0 if reached else malus
        self.last_observation = observation

        return observation, reward, reached, {"collided": collided, "reached": reached}

    def is_reachable(self, positions):
        """
        Check if all positions are reachable

        Args:
            positions: a numpy array of positions. Could be a batch of positions of the shape (n, 2) for a batch of
                size n, or a numpy array of shape (2,) for a single position.

        Returns:
            If the given position(s) are reachable or not.

        """
        if isinstance(positions, (tuple, list)):
            positions = np.array(positions)
        assert isinstance(positions, np.ndarray) and positions.shape[-1] == 2

        result = (self.maze_info["x_max"] >= positions[..., 0] >= self.maze_info["x_min"]
                  and self.maze_info["y_max"] >= positions[..., 1] >= self.maze_info["y_min"])
        for obstacle in self.maze_info["obstacles"]:
            if obstacle["type"] == "rectangle":
                obstacle_x = float(obstacle["position"].split(" ")[0])
                obstacle_y = float(obstacle["position"].split(" ")[1])
                size_x = float(obstacle["size"].split(" ")[0])
                size_y = float(obstacle["size"].split(" ")[1])
                box = Box(low=np.array([obstacle_x - size_x / 2, obstacle_y - size_y / 2]).astype(np.float32),
                          high=np.array([obstacle_x + size_x / 2, obstacle_y + size_y / 2]).astype(np.float32))
                result = result and not box.contains(positions.astype(box.dtype))
                if not result:
                    break
            else:
                raise NotImplementedError()
        return result

    def set_goal(self, new_goal):
        if isinstance(new_goal, (tuple, list)):
            new_goal = np.ndarray(new_goal)
        assert isinstance(new_goal, np.ndarray) and new_goal.shape[-1] == 2

        self.goal = new_goal

        if not self.real_world:
            # Move the goal marker to the goal position, so we can see the goal in the gazebo simulation:
            request = SetEntityState.Request()
            request.state.name = "goal_marker"
            request.state.pose = Pose(position=Point(x=self.goal[0].item(), y=self.goal[1].item(), z=0.2))
            self.set_entity_state_client.call_async(request)

    def hide_marker(self):
        request = SetEntityState.Request()
        request.state.name = "goal_marker"
        request.state.pose = Pose(position=Point(z=-1.))
        self.set_entity_state_client.call_async(request)

    def show_marker(self):
        request = SetEntityState.Request()
        request.state.name = "goal_marker"
        request.state.pose = Pose(position=Point(z=0.2))
        self.set_entity_state_client.call_async(request)

    def place_marker(self, x, y):
        request = SetEntityState.Request()
        request.state.name = "goal_marker"
        request.state.pose = Pose(position=Point(x=x, y=y, z=0.2))
        self.set_entity_state_client.call_async(request)

    def reset(self):
        if self.real_world:
            self.commands_publisher.publish(Twist())  # Make sure the agent don't move
            # print("\n\n")
            # print("    THE ROBOT IS STUCK AND REQUIRE A RESET.   ")
            # input("    PLACE THE ROBOT TO ITS INITIAL POSITION AND PRESS ANY KEY WHEN ITS DONE.")
            # print("\n\n")

            # Fetch observation, until it match with the expected start position (to make sure the world reset request
            # above have been executed)
            observation = None
            while observation is None:
                observation, _ = self.get_observation()

            goal = random.choice(self.available_goals)
        else:
            self.reset_client.call_async(Empty.Request())
            time.sleep(0.08)  # Wait for the environment to reset

            # Expected start position
            start_sim_position = np.array(self.maze_info["robot_reset_position"].split(" ")[:2]).astype(np.float32)

            # Fetch observation, until it match with the expected start position (to make sure the world reset request
            # above have been executed)
            observation = None
            while observation is None or (np.abs(observation[:2] - np.array(start_sim_position)) > 0.01).all():
                observation, _ = self.get_observation()

            # Sample a goal position from reachable tiles
            goal = None
            while goal is None or not self.is_reachable(goal):
                goal = self.goal_space.sample()

            # self.environment_size_scale / 2 aka: half the size of a maze_array tile.
            # In other words, we sample a goal uniformly inside the tile we sampled.

        self.set_goal(goal)

        return observation, self.goal

    ##########################################################################
    ####                       SUBSCRIBERS CALLBACKS                      ####
    ##########################################################################
    def _lidar_callback(self, msg):
        self.last_lidar_message = msg
        self.waiting_for_lidar = False

    def _position_callback(self, msg):
        self.last_position_message = msg
        self.waiting_for_position = False

    def get_goal_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[:2]

    def pause_simulation(self):
        if self.real_world:
            self.commands_publisher.publish(Twist())
        else:
            self.pause_client.call_async(Empty.Request())

    def unpause_simulation(self):
        if self.real_world:
            vel_cmd = Twist()
            vel_cmd.linear.x = self.velocity[0].item()
            vel_cmd.angular.z = self.velocity[1].item()
            self.commands_publisher.publish(vel_cmd)
        else:
            self.unpause_client.call_async(Empty.Request())

    def render(self, resolution=100, ignore_goal=False) -> np.ndarray:
        """
        :params pixels_tiles_width: int, tiles width in pixel.
        :params ignore_goal: bool, whether to draw goal on the image or not.
        Return a np.ndarray of size (width, height, 3) of pixels that represent the environments and it's walls
        :return: The final image.
        """

        image_width_px, image_height_px = (self.goal_space.high - self.goal_space.low)[:2].astype(np.int64) * resolution
        image = np.full((image_height_px, image_width_px, 3), 255, dtype=np.uint8)

        for obstacle in self.maze_info["obstacles"]:
            if obstacle["type"] == "rectangle":
                obstacle_pos = np.array(obstacle["position"].split(" ")[:2]).astype(np.float32)
                obstacle_size = np.array(obstacle["size"].split(" ")[:2]).astype(np.float32) / 2

                shape_i_min = (image_height_px -
                               int((obstacle_pos[1] + obstacle_size[1] - self.goal_space.low[1]) * resolution))
                shape_i_max = (image_height_px -
                               int((obstacle_pos[1] - obstacle_size[1] - self.goal_space.low[1]) * resolution))
                shape_j_min = int((obstacle_pos[0] - obstacle_size[0] - self.goal_space.low[0]) * resolution)
                shape_j_max = int((obstacle_pos[0] + obstacle_size[0] - self.goal_space.low[0]) * resolution)
                image[shape_i_min:shape_i_max, shape_j_min:shape_j_max, :] = TileType.WALL.value
            else:
                raise NotImplementedError()

        if not ignore_goal:
            self.place_point(*self.goal, image=image, color=[255, 0, 0], width=10)

        ##############
        # DRAW AGENT
        ##############
        if self.last_observation is None:
            self.step(np.array([0, 0]))

        agent_color = [0, 0, 255]
        size_ratio = 0.1

        # Place a point at agent's position
        self.place_point(*self.last_observation[:2], image=image, color=agent_color, width=int(resolution * size_ratio))

        # Draw orientation line
        orientation = quaternion_to_euler(self.last_observation[3:7])
        position_1 = self.last_observation[:2]
        position_2 = position_1 + np.array([math.cos(orientation[-1]), math.sin(orientation[-1])]) * (size_ratio * 2)
        start_center_pixel_x, start_center_pixel_y = self.get_image_pixel_from_position(*position_1, image)
        stop_center_pixel_x, stop_center_pixel_y = self.get_image_pixel_from_position(*position_2, image)

        try:
            rr, cc, val = line_aa(start_center_pixel_x, start_center_pixel_y, stop_center_pixel_x, stop_center_pixel_y)
            old = image[rr, cc]
            extended_val = np.tile(val, (3, 1)).T
            image[rr, cc] = (1 - extended_val) * old + extended_val * agent_color
        except:
            pass
        return image

    def place_point(self, x: float, y: float, image: np.ndarray, color: Union[None, list, np.ndarray] = None,
                    width: int = 5):
        if color is None:
            color = [0, 0, 0]
        assert isinstance(image, np.ndarray) and image.shape[-1] == 3 and len(image.shape) == 3
        center_i, center_j = self.get_image_pixel_from_position(x, y, image)
        for i in range(center_i - width, center_i + width):
            for j in range(center_j - width, center_j + width):
                dist = math.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
                if dist < width and 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                    image[i, j] = color
        return image

    def get_image_pixel_from_position(self, x, y, image: np.ndarray):
        image_height, image_width = image.shape[:2]
        goal_space_dim = (self.goal_space.high - self.goal_space.low)[:2].astype(np.int64)
        resolution = np.array([image_width, image_height]) / goal_space_dim
        assert resolution[0] == resolution[1]
        resolution = resolution[0]

        # Let's compute pixel (i, j) coordinates such that image[i][j] is the pixel that belongs to the x, y coordinates
        pixel_j = int((x - self.goal_space.low[0]) * resolution)
        pixel_i = image_height - int((y - self.goal_space.low[1]) * resolution)
        return pixel_i, pixel_j

    def kill_gazebo(self):
        self.gazebo_subprocess.terminate()
        first_try_time = time.time()
        while self.gazebo_subprocess.is_alive():
            if time.time() - first_try_time > 1:
                self.gazebo_subprocess.kill()

    def __del__(self):
        self.kill_gazebo()
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    env = SingleRobotEnvironmentV2(map_name=RobotMapsIndex.EMPTY.value, use_lidar=True, use_odometry=True,
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
