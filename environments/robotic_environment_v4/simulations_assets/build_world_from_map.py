import math
import pathlib
import numpy as np
import importlib
from lxml import etree
from typing import Union


def load_walls(maze_array: Union[list, np.ndarray], scale: float):
    maze_array = np.array(maze_array)

    maze_dims = np.array(maze_array.shape) * scale
    x_max, y_max = (maze_dims / 2).tolist()
    x_min, y_min = -x_max, -y_max
    maze_info = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "obstacles": []}
    try:
        start = np.argwhere(maze_array == 2)[0]
    except IndexError:
        start = np.argwhere(maze_array == 0)[0]
    start = (start + 0.5) * scale
    robot_start_x, robot_start_y = start[1] + x_min, - (start[0] + y_min)
    maze_info["robot_reset_position"] = "{0} {1} 0.000929 0 0 0.803909".format(robot_start_x, robot_start_y)

    horizontal_wall_id = 0
    vertical_wall_id = 0
    while np.isin(1, maze_array):
        first_tile = np.argwhere(maze_array == 1)[0].copy().tolist()

        # build the vertical wall
        vertical_wall = [first_tile]
        current_i_index = first_tile[0]  # Add wall parts at the left
        while current_i_index - 1 > 0 and maze_array[current_i_index - 1, first_tile[1]].item() == 1:
            vertical_wall.insert(0, [current_i_index - 1, first_tile[1]])
            current_i_index -= 1
        current_i_index = first_tile[0]  # Add wall parts at the right
        while current_i_index + 1 < len(maze_array) \
                and maze_array[current_i_index + 1, first_tile[1]].item() == 1:
            vertical_wall.append([current_i_index + 1, first_tile[1]])
            current_i_index += 1

        # build the horizontal wall
        horizontal_wall = [first_tile]
        current_j_index = first_tile[1]  # Add wall parts at the left
        while current_j_index - 1 > 0 and maze_array[first_tile[0], current_j_index - 1].item() == 1:
            horizontal_wall.insert(0, [first_tile[0], current_j_index - 1])
            current_j_index -= 1
        current_j_index = first_tile[1]  # Add wall parts at the right
        while current_j_index + 1 < len(maze_array[1]) \
                and maze_array[first_tile[0], current_j_index + 1].item() == 1:
            horizontal_wall.append([first_tile[0], current_j_index + 1])
            current_j_index += 1

        # Choose the longer wall and add it to walls list.
        if len(horizontal_wall) > len(vertical_wall):
            # replace every tile in the chosen wall by 0, so it will not be added again
            row_index = horizontal_wall[0][0]
            col_from, col_to = horizontal_wall[0][1], horizontal_wall[-1][1]
            maze_array[row_index, col_from:col_to + 1] = 0

            # Add this wall to the memory
            center_i = (horizontal_wall[-1][0] + horizontal_wall[0][0] + 1) / 2
            center_j = (horizontal_wall[-1][1] + horizontal_wall[0][1] + 1) / 2
            position_x, position_y = center_j * scale + x_min, - (center_i * scale + y_min)
            width = (horizontal_wall[-1][1] - horizontal_wall[0][1] + 1) * scale

            obstacle = {
                "name": "horizontal_wall_" + str(horizontal_wall_id),
                "type": "rectangle",
                "position": str(position_x) + " " + str(position_y),
                "size": str(width) + " " + str(scale)
            }
            maze_info["obstacles"].append(obstacle)
            horizontal_wall_id += 1
        else:
            # replace every tile in the chosen wall by 0, so it will not be added again
            col_index = vertical_wall[0][1]
            row_from, row_to = vertical_wall[0][0], vertical_wall[-1][0]
            maze_array[row_from:row_to + 1, col_index] = 0

            # Add this wall to the memory
            center_i = (vertical_wall[-1][0] + vertical_wall[0][0] + 1) / 2
            center_j = (vertical_wall[-1][1] + vertical_wall[0][1] + 1) / 2
            position_x, position_y = center_j * scale + x_min, - (center_i * scale + y_min)

            height = (vertical_wall[-1][0] - vertical_wall[0][0] + 1) * scale

            obstacle = {
                "name": "vertical_wall_" + str(vertical_wall_id),
                "type": "rectangle",
                "position": str(position_x) + " " + str(position_y),
                "size": str(scale) + " " + str(height)
            }
            maze_info["obstacles"].append(obstacle)
            vertical_wall_id += 1
    return maze_info


def maze_pos_to_simulation_pos(i, j, scale, maze_array):
    """
    returns the expected position inside the simulation from the coordinate of a tile in the maze_array
    """
    shape = np.array(maze_array).shape
    simu_x = - (i - (shape[0] / 2 - 0.5)) * scale
    simu_y = - (j - (shape[1] / 2 - 0.5)) * scale
    return simu_x, simu_y


def simulation_pos_to_maze_pos(simu_x, simu_y, scale, maze_array):
    """
    returns the coordinate (as float) inside the maze_array from a position inside the simulation
    """
    shape = np.array(maze_array).shape
    i = - simu_x / scale + (shape[0] / 2 - 0.5)
    j = - simu_y / scale + (shape[1] / 2 - 0.5)
    return i, j


def generate_xml(map_name: str, robot_name: str, scale: float = 1., walls_height: float = 0.5, nb_markers: int = 0) -> \
        (dict, str):
    """
    Generate a gazebo environment from a template and maze walls description.
    :params map_name: str, name of the map, used to find the maze array with the walls description.
    :params robot_name: str, name of the robot, used to find the robot xml.
    :params scale: float, scaling factor of the maze (aka. width of a maze's tile in the simulation).
    :params walls_height: float, height of walls in the simulation.
    :params nb_markers: int, default=0, how many marker should be added in the .world file. Markers are static sphere
        models without collision, used to mark a position like a goal or an RRT graph node for example.
    :returns: (as a tuple of two elements)
     - A 'dict' that contains maze description file information.
     - A 'str' that correspond to the path where the resultant xml file has been generated.
    """

    walls_height = str(walls_height)

    # Load base ant-maze file etree:

    # Setup paths
    worlds_directory_path = pathlib.Path(__file__).parent / "worlds"
    maps_directory_path = pathlib.Path(__file__).parent / "maps"
    robots_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "robots"
                           / (robot_name + ".xml"))
    block_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "block.xml")
    sphere_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "sphere.xml")
    camera_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "camera.xml")
    template_file_path = str(worlds_directory_path / "template.world")

    tree = etree.parse(template_file_path)

    # Find 'world' in template.world xml file:
    world_node = None
    for child in tree.getroot():
        if child.tag == "world":
            world_node = child
            break
    else:
        class XmlParseError(Exception):
            pass

        raise XmlParseError("Node 'world' not found in " + template_file_path + ".")

    # Load maps data
    try:
        if __name__ != "__main__":
            import_path = ".".join(__name__.split(".")[:-1]) + ".maps." + map_name
        else:
            import_path = "environments.robotic_environment_v2.simulations_assets.maps." + map_name
        try:
            maze_array = np.array(importlib.import_module(import_path).maze_array)
            try:
                scale = importlib.import_module(import_path).scale
            except AttributeError:
                pass
        except AttributeError as e:
            maze_info = importlib.import_module(import_path).maze_info
    except ModuleNotFoundError:
        raise FileNotFoundError("Could not find map with name " + map_name + ".py, Please verify it is located in "
                                + str(maps_directory_path) + ".")

    def add_wall(name, position, size):
        block_node = etree.parse(block_file_path).getroot()
        block_node.attrib["name"] = name

        # Set position
        block_node.find("pose").text = position + " 0 0 0"
        # Set size
        block_node.find("link").find("collision").find("geometry").find("box").find("size").text = size
        block_node.find("link").find("visual").find("geometry").find("box").find("size").text = size

        world_node.append(block_node)

    if "maze_array" in locals():
        # Load walls
        maze_info = load_walls(maze_array, scale)

    assert "maze_info" in locals() and maze_info is not None

    # Compute some info
    maze_width = maze_info["x_max"] - maze_info["x_min"]
    maze_height = maze_info["y_max"] - maze_info["y_min"]
    maze_center_x = (maze_info["x_max"] + maze_info["x_min"]) / 2
    maze_center_y = (maze_info["y_max"] + maze_info["y_min"]) / 2
    walls_z_position = str(float(walls_height) / 2)

    if not "maze_array" in locals():
        # Place external walls
        position = "{0} {1} " + walls_z_position
        size = "{0} {1} " + walls_height
        walls_width = 1
        add_wall(name="vertical_border_wall_1",
                 position=position.format(str(maze_info["x_min"] - float(walls_width) / 2), str(maze_center_y)),
                 size=size.format(walls_width, str(maze_height + float(walls_width) * 2)))
        add_wall(name="vertical_border_wall_2",
                 position=position.format(str(maze_info["x_max"] + float(walls_width) / 2), str(maze_center_y)),
                 size=size.format(walls_width, str(maze_height + float(walls_width) * 2)))
        add_wall(name="horizontal_border_wall_1",
                 position=position.format(str(maze_center_x), str(maze_info["y_min"] - float(walls_width) / 2)),
                 size=size.format(str(maze_width + float(walls_width) * 2), walls_width))
        add_wall(name="horizontal_border_wall_2",
                 position=position.format(str(maze_center_x), str(maze_info["y_max"] + float(walls_width) / 2)),
                 size=size.format(str(maze_width + float(walls_width) * 2), walls_width))

    for obstacle in maze_info["obstacles"]:
        if obstacle["name"][-1] == "6":
            debug = 1
        if obstacle["type"] == "rectangle":
            position_list = obstacle["position"].split(" ")
            if len(position_list) == 2:
                position_list.append(walls_z_position)

            size_list = obstacle["size"].split(" ")
            if len(size_list) == 2:
                size_list.append(walls_height)

            add_wall(name=obstacle["name"],
                     position=" ".join(position_list),
                     size=" ".join(size_list))
        else:
            raise NotImplementedError("This obstacle type is not implemented yet.")

    # Add camera
    camera_node = etree.parse(camera_file_path).getroot()
    cam_pose_node = camera_node.find("camera").find("pose")
    # cam_pose_node.attrib["relative_to"] = "horizontal_wall_1"
    # cam_pose_node.text = "0 -10 8 0 0.7 " + str(math.pi / 2)
    cam_pose_node.text = "{0} {1} {2} -1.57 1.57 0".format(maze_center_x, maze_center_y, maze_height * 2)
    world_node.append(camera_node)

    # Load robot xml
    robot_node = etree.parse(robots_file_path).getroot()
    # Get default robot position
    # Set robot position
    pose = robot_node.find("pose")
    pose_list = pose.text.split(" ")
    pose_list[-1] = str(-0.5 * math.pi)
    pose.text = maze_info["robot_reset_position"]
    # Spawn robot
    world_node.append(robot_node)

    # Add goal_marker
    sphere_node = etree.parse(sphere_file_path).getroot()
    sphere_node.attrib["name"] = "goal_marker"
    sphere_node.find("link").find("visual").find("material").find("ambient").text = "1 0 0 1"
    world_node.append(sphere_node)

    # Generate output file from etree information
    xml_output_path = worlds_directory_path / ("generated_" + map_name + ".world")
    tree.write(str(xml_output_path))

    # Make generated file more pretty
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_output_path, parser)
    tree.write(xml_output_path, pretty_print=True)
    return maze_info, xml_output_path
