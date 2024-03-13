import time

import numpy as np
from enum import Enum


def get_dict_as_str(input_dict, start_tab="", tab="    "):
    result = ""
    for k, v in input_dict.items():
        if result != "":
            result += ",\n"
        if isinstance(v, dict):
            result += get_dict_as_str(v, start_tab + tab)
        if isinstance(v, list):
            result += start_tab + tab + str(k) + ": [" + "\n"
            for index, elt in enumerate(v):
                if isinstance(elt, dict):
                    result += get_dict_as_str(elt, start_tab + tab + tab)
                else:
                    result += start_tab + tab + tab + str(elt)
                result += "\n" if index == len(v) - 1 else ",\n"
            result += start_tab + tab + "]"
        else:
            if isinstance(v, str):
                result += start_tab + tab + str(k) + ": \"" + v + "\""
            else:
                result += start_tab + tab + str(k) + ": " + str(v) + ""
    return start_tab + "{\n" + result + "\n" + start_tab + "}"


def get_quaternion_from_euler(roll, pitch, yaw) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qx, qy, qz, qw


def get_euler_from_quaternion(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    euler_x = np.arctan2(t0, t1)

    t2 = 2. * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    euler_y = np.arcsin(t2)

    t3 = 2. * (w * z + x * y)
    t4 = 1. - 2. * (y * y + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def get_point_image_after_rotation(x, y, angle, rotation_center_x=0., rotation_center_y=0.):
    """
    Compute and return the coordinates of a point p', which is the image of the point (x, y) after a rotation of a given
    "angle" (in gradiant) around the given center.
    @param x: X coordinate of the rotated point,
    @param y: Y coordinate of the rotated point,
    @param angle: rotation angle in gradiant,
    @param rotation_center_x: Rotation center's X coordinates,
    @param rotation_center_y: Rotation center's Y coordinates.
    """

    result_x = np.cos(angle) * (x - rotation_center_x) - np.sin(angle) * (y - rotation_center_y) + rotation_center_x
    result_y = np.cos(angle) * (y - rotation_center_y) + np.sin(angle) * (x - rotation_center_x) + rotation_center_y
    return result_x, result_y


def print_replace_above(n_lines_above, message):
    assert isinstance(n_lines_above, int) and n_lines_above >= 0
    if n_lines_above > 0:
        print("\033[" + str(n_lines_above) + "A\033[K", end="")
    print(message)
    if n_lines_above > 1:
        print("\033[" + str(n_lines_above - 1) + "B", end="")


if __name__ == "__main__":
    messages = ["A", "B", "C", "D"]
    for m in messages:
        print(m)

    for num in range(20):
        for i, m in enumerate(messages):
            print_replace_above(4 - i, m + str(num))
        time.sleep(.3)
