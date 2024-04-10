import time
from enum import Enum
import random
from multiprocessing import Process
from launch_gazebo import launch_gazebo
import numpy as np
from gym.spaces import Box
from std_srvs.srv import Empty
from environments.robotic_environment.simulations_assets.build_world_from_map import generate_xml
from environments.goal_conditioned_environment import GoalConditionedEnvironment
import rclpy
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile


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
                 map_name: str = RobotMapsIndex.FOUR_ROOMS.value,
                 robot_name: str = RobotsIndex.IROBOT.value,
                 environment_size_scale: float = 0.5,

                 # Environment behaviour settings
                 real_time: bool = True,
                 simulation_step_duration: float = 0.1,  # UNUSED if real_time=True
                 goal_reachability_threshold: float = 0.40,
                 collision_distance_threshold: float = 0.30,
                 reward_at_collision: float = None,          # Set the reward given at collision.
                 reward_once_reached: float = 0,
                 sparse_reward: bool = True,

                 max_velocity: np.ndarray = np.array([0.5, 1]),
                 max_action: np.ndarray = np.array([0.2, 0.4]),

                 # State composition and settings
                 use_lidar: bool = False,
                 lidar_max_angle: float = None,     # USELESS if variable use_lidar is false
                 nb_lidar_beams: int = 20,          # USELESS if variable use_lidar is false
                 use_odometry: bool = False,  # Use odometry topic as position/orientation (instead of /pose by default)
                 ):

        """
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
        self.maze_array, world_file_path = generate_xml(map_name, robot_name, scale=self.environment_size_scale)

        # Launch gazebo using the launch file and the robot
        p = Process(target=launch_gazebo, args=(str(world_file_path),))  # TODO add robot sdf file, maybe later
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
                    self.reset_service_name = service[0]
                    assert service[1][0] == "std_srvs/srv/Empty", "Wrong type for reset service."
                    print("Found gazebo service {}".format(service[0]))
                    break
            time.sleep(0.5)

        # Look for topics
        lidar_message_type = "sensor_msgs/msg/LaserScan"
        position_message_type = "nav_msgs/msg/Odometry" if self.use_odometry else "geometry_msgs/msg/PoseStamped"
        commands_message_type = "geometry_msgs/msg/Twist"
        for topic in self.node.get_topic_names_and_types():
            topic_name = topic[0]
            topic_type = topic[1][0]
            if self.use_lidar and lidar_message_type == topic_type:
                self.lidar_topic_name = topic_name
            elif position_message_type == topic_type:
                self.position_topic_name = topic_name
            elif commands_message_type == topic_type:
                self.commands_topic_name = topic_name

        # Find missing topics
        if self.use_lidar and not hasattr(self, "lidar_topic_name"):
            raise EnvironmentError("No topic found with type '" + lidar_message_type + "' alone for lidar.")
        if not hasattr(self, "position_topic_name"):  # MANDATORY
            raise EnvironmentError("No topic found with type '" + position_message_type + "' alone for position.")
        if not hasattr(self, "commands_topic_name"):  # MANDATORY
            raise EnvironmentError("No topic found with type '" + commands_message_type + "' alone for commands.")

        # Find required service names
        for service in self.node.get_service_names_and_types():
            service_name = service[0]
            service_type = service[1][0]
            if service_type == "std_srvs/srv/Empty":
                if not self.real_time and service_name.split("/")[-1].startswith("pause"):
                    self.pause_service_name = service_name
                elif not self.real_time and service_name.split("/")[-1].startswith("unpause"):
                    self.unpause_service_name = service_name
                if service_name.split("/")[-1] == "reset_world":
                    self.reset_service_name = service_name

            if not self.real_time and not hasattr(self, "pause_service_name"):
                raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service pause.")
            if not self.real_time and not hasattr(self, "unpause_service_name"):
                raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service unpause.")
            if not hasattr(self, "reset_service_name"):
                raise EnvironmentError("No service found with type 'std_srvs/srv/Empty' alone for service reset_world.")

        # Setup subscribers
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST,
                                depth=1)

        # - Setup lidar subscriber
        if self.use_lidar:
            self.last_lidar_message = None
            self.lidar_subscriber = self.node.create_subscription(LaserScan, self.lidar_topic_name,
                                                                  self._lidar_callback, qos_profile=qos_sensor)

        # - Setup position subscriber
        self.last_position_message = None
        self.position_message_type = Odometry if self.use_odometry else PoseStamped
        self.position_subscriber = self.node.create_subscription(self.position_message_type, self.position_topic_name,
                                                                 self._position_callback, qos_profile=qos_sensor)

        # Setup command publisher
        self.commands_publisher = self.node.create_publisher(Twist, self.commands_topic_name, QoSProfile(depth=10))

        # Setup services clients
        if not self.real_time:
            self.unpause_client = self.node.create_client(Empty, self.unpause_service_name)
            self.pause_client = self.node.create_client(Empty, self.pause_service_name)
        self.reset_client = self.node.create_client(Empty, self.reset_service_name)

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
        high = np.full(self.observation_size, float("inf"), dtype=np.float16)
        self.observation_space = Box(-high, high, dtype=np.float16)
        self.goal_space = Box(- high[:2], high[:2], dtype=np.float16)
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
            print("Waiting for lidar topic ...")
            while self.last_lidar_message is None:
                rclpy.spin_once(self.node)
                time.sleep(0.1)

    def step(self, action):

        #### PERFORM ACTION ####

        # Publish the robot action
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

        #### GET STATE ####

        self.waiting_for_position = True
        self.waiting_for_lidar = True
        while self.waiting_for_position and self.waiting_for_lidar:
            rclpy.spin_once(env.node)

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
        collided = False
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

        # Compute distance to the goal, and compute reward from it
        distance = np.linalg.norm(agent_observation[:2] - self.goal)
        reached = distance < self.goal_reachability_threshold
        malus = -1 if self.sparse_reward else - distance
        reward = 0 if reached else malus

        return agent_observation, reward, reached, {"collided": collided}

    def map_coordinates_to_env_position(self, x, y):
        x = self.maze_array.shape[1] + x if x < 0 else x
        y = self.maze_array.shape[0] + y if y < 0 else y
        return np.array([x - self.maze_array.shape[1] / 2 + 0.5, - (y - self.maze_array.shape[0] / 2 + 0.5)])

    def reset(self):
        self.reset_client.call_async(Empty.Request())

        # setup goal
        goal_tile = np.flip(random.choice(np.argwhere(self.maze_array != TileType.WALL.value)))
        goal_position = self.map_coordinates_to_env_position(*goal_tile)
        self.goal = np.random.uniform(goal_position - 0.5, goal_position + 0.5)

        pass

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


if __name__ == "__main__":
    env = SingleRobotEnvironment(map_name="medium_maze", use_lidar=True, use_odometry=True)

    for episode_id in range(10):
        print("Episode {}".format(episode_id))
        env.reset()

        # for step_id in range(500):
        #     print("Step {}".format(step_id))
        #     env.step(env.action_space.sample())
    