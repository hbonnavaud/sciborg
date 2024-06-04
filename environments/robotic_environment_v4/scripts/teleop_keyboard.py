import os
import select
import sys
import rclpy
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile
import numpy as np


robot_namespace = "/irobot02"
robot_namespace = ""


if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty


if robot_namespace.startswith("/Turtle"):
    # Choose the one according to the turtle type
    TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']

    BURGER_MAX_LIN_VEL = 0.22
    BURGER_MAX_ANG_VEL = 2.84

    WAFFLE_MAX_LIN_VEL = 0.26
    WAFFLE_MAX_ANG_VEL = 1.82

    MAX_LIN_VEL = BURGER_MAX_LIN_VEL
    MAX_ANG_VEL = BURGER_MAX_ANG_VEL
else:
    MAX_LIN_VEL = 1
    MAX_ANG_VEL = 1

LIN_VEL_STEP_SIZE = 1
ANG_VEL_STEP_SIZE = 0.4
LIN_VEL_SLOPE = LIN_VEL_STEP_SIZE
ANG_VEL_SLOPE = ANG_VEL_STEP_SIZE

forward_key = 'z'
left_key = 'q'
right_key = 'd'
backward_key = 's'
stop_key = 'x'  # + space

msg = """
Control Your robot!
---------------------------
Moving around:
        $f
   $l    $b    $r
        $s

$f/$b : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
$l/$r : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, $s : force stop

CTRL-C to quit
""".replace("$f", forward_key).replace("$l", left_key).replace("$r", right_key).replace("$b", backward_key).replace("$s", stop_key)

e = """
Communications Failed
"""


def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_vels(target_linear_velocity, target_angular_velocity):
    print('currently:\tlinear velocity {0}\t angular velocity {1} '.format(
        target_linear_velocity,
        target_angular_velocity))


def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output


forward_key = 'z'
left_key = 'q'
right_key = 'd'
backward_key = 's'
stop_key = 'x'  # + space


def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()

    node = rclpy.create_node('teleop_keyboard')
    qos = QoSProfile(depth=10)
    pub = node.create_publisher(Twist, '/irobot02/cmd_vel', qos)

    status = 0
    target_linear_velocity = 0.0
    target_angular_velocity = 0.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0

    try:
        print(msg)
        while True:
            key = get_key(settings)
            if key == forward_key:
                target_linear_velocity = np.clip(target_linear_velocity + LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == backward_key:
                target_linear_velocity = np.clip(target_linear_velocity - LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == left_key:
                print("LEFT LEFT ")
                target_angular_velocity = np.clip(target_angular_velocity + ANG_VEL_STEP_SIZE, -MAX_ANG_VEL, MAX_ANG_VEL)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == right_key:
                target_angular_velocity = np.clip(target_angular_velocity - ANG_VEL_STEP_SIZE, -MAX_ANG_VEL, MAX_ANG_VEL)
                status = status + 1
                print_vels(target_linear_velocity, target_angular_velocity)
            elif key == ' ' or key == stop_key:
                target_linear_velocity = 0.0
                control_linear_velocity = 0.0
                target_angular_velocity = 0.0
                control_angular_velocity = 0.0
                print_vels(target_linear_velocity, target_angular_velocity)
            else:
                if key == '\x03':
                    break

            # odom_sub = rospy.Subscriber('/odom', Odometry, position_callback)
            if status == 20:
                print(msg)
                status = 0

            twist = Twist()

            control_linear_velocity = make_simple_profile(control_linear_velocity, target_linear_velocity,
                                                          LIN_VEL_SLOPE)

            twist.linear.x = control_linear_velocity

            control_angular_velocity = make_simple_profile(control_angular_velocity, target_angular_velocity,
                                                           ANG_VEL_SLOPE)

            twist.angular.z = control_angular_velocity

            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
