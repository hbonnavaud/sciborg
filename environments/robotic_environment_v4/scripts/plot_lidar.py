#! /usr/bin/env python
import sys
import rclpy
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import sched, time
import matplotlib.pyplot as plt
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from math import cos, sin, pi


if len(sys.argv) == 1:
	turtle_id = "1"
else:
	turtle_id = sys.argv[1]


def callback(msg):
	last_angle = msg.angle_min
	points = []
	for r in msg.ranges:
		if msg.range_min < r < msg.range_max:
			print("angle=", last_angle, ", r=", r)
			points.append([cos(last_angle) * r, sin(last_angle) * r])
		last_angle += msg.angle_increment
	points = np.array(points)
	plt.cla()
	plt.scatter(points[:, 0], points[:, 1])
	plt.pause(0.00001)


plt.close(plt.figure(1))
plt.ion()

rclpy.init()
node = rclpy.create_node('check_scan')

qos_sensor = QoSProfile(
	reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
	history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
	depth=1
)

lazer_scan_sub = node.create_subscription(LaserScan, '/demo/laser/out', callback, qos_profile=qos_sensor)
rclpy.spin(node)
