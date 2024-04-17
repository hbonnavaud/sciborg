import time
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation


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


def euler_to_quaternion(euler_angle):
    return Rotation.from_euler("xyz", euler_angle).as_quat()


def quaternion_to_euler(quaternion):
    return Rotation.from_quat(quaternion).as_euler('xyz')


def are_euler_angles_equivalent(euler_angle_1, euler_angle_2, tolerance=np.deg2rad(1)):
    dot_product = np.dot(euler_to_quaternion(euler_angle_1),
                         euler_to_quaternion(euler_angle_2))
    angle_diff = 2 * np.arccos(np.abs(dot_product))
    return angle_diff <= tolerance


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

    def angles_conversion_tests():
        angles = np.array([
            [0, 0, 0],
            [np.pi / 4, np.pi / 6, np.pi / 3],
            [np.pi / 2, np.pi / 4, np.pi / 2],
            [np.pi, np.pi / 3, np.pi / 6],
            [3 * np.pi / 4, np.pi / 2, np.pi],
            [np.pi / 6, np.pi / 2, np.pi / 4],
            [np.pi / 3, np.pi / 6, 3 * np.pi / 4],
            [np.pi / 2, 0, np.pi],
            [0, np.pi / 2, np.pi],
            [np.pi / 4, np.pi / 4, np.pi / 4]
        ])

        angles_as_quat = euler_to_quaternion(angles)
        angles_as_euler = quaternion_to_euler(angles_as_quat)
        for i in range(len(angles)):
            angle = np.around(angles[i], 2)
            angle_as_quat = np.around(angles_as_quat[i], 2)
            angle_as_euler = np.around(angles_as_euler[i], 2)
            t1 = "Initial: " + str(angle)
            t2 = "quat: " + str(angle_as_quat)
            t3 = "euler: " + str(angle_as_euler)
            equals = (angle == angle_as_euler).all() or are_euler_angles_equivalent(angle, angle_as_euler)
            t4 = "equal: " + str(equals)
            t2 = t1 + " " * max(0, 30 - len(t1)) + t2
            t3 = t2 + " " * max(0, 65 - len(t2)) + t3
            t4 = t3 + " " * max(0, 90 - len(t3)) + t4
            print(t4)


    def print_replace_above_test():
        messages = ["A", "B", "C", "D"]
        for m in messages:
            print(m)

        for num in range(20):
            for i, m in enumerate(messages):
                print_replace_above(4 - i, m + str(num))
            time.sleep(.3)
