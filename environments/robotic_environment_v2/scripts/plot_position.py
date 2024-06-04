#! /usr/bin/env python

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import sched, time
import matplotlib.pyplot as plt
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from argparse import ArgumentParser

plt.close(plt.figure(1))
plt.ion()

positions = []
max_size = 100000


def callback(msg):
	position = msg.pose.position
	print("Received ", position.x, "; ", position.y)
	if len(positions) > max_size:
		positions.pop(0)
	positions.append((position.x, position.y))
	data = np.array(positions)
	plt.cla()
	plt.scatter(data[:, 0], data[:, 1])
	plt.pause(0.00001)


rclpy.init()
node = rclpy.create_node('check_odometry')

qos_sensor = QoSProfile(
	reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
	history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
	depth=1
)

odom_sub = node.create_subscription(PoseStamped, '/vrpn_mocap/IRobot_02/pose', callback, qos_profile=qos_sensor)
rclpy.spin(node)
