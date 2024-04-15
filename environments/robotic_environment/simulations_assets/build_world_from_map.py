import inspect
import math
import pathlib
import numpy as np
import importlib
import os.path
from lxml import etree
from lxml.builder import E


def load_walls(maze_array):
    horizontal_walls = []
    vertical_walls = []
    maze_array_np = np.array(maze_array)
    while np.isin(1, maze_array_np):
        first_tile = np.argwhere(maze_array_np == 1)[0].copy().tolist()

        # build the vertical wall
        vertical_wall = [first_tile]
        current_i_index = first_tile[0]  # Add wall parts at the left
        while current_i_index - 1 > 0 and maze_array_np[current_i_index - 1, first_tile[1]].item() == 1:
            vertical_wall.insert(0, [current_i_index - 1, first_tile[1]])
            current_i_index -= 1
        current_i_index = first_tile[0]  # Add wall parts at the right
        while current_i_index + 1 < len(maze_array) \
                and maze_array_np[current_i_index + 1, first_tile[1]].item() == 1:
            vertical_wall.append([current_i_index + 1, first_tile[1]])
            current_i_index += 1

        # build the horizontal wall
        horizontal_wall = [first_tile]
        current_j_index = first_tile[1]  # Add wall parts at the left
        while current_j_index - 1 > 0 and maze_array_np[first_tile[0], current_j_index - 1].item() == 1:
            horizontal_wall.insert(0, [first_tile[0], current_j_index - 1])
            current_j_index -= 1
        current_j_index = first_tile[1]  # Add wall parts at the right
        while current_j_index + 1 < len(maze_array[1]) \
                and maze_array_np[first_tile[0], current_j_index + 1].item() == 1:
            horizontal_wall.append([first_tile[0], current_j_index + 1])
            current_j_index += 1

        # Choose the longer wall and add it to walls list.
        if len(horizontal_wall) > len(vertical_wall):
            # replace every tile in the chosen wall by 0, so it will not be added again
            row_index = horizontal_wall[0][0]
            col_from, col_to = horizontal_wall[0][1], horizontal_wall[-1][1]
            maze_array_np[row_index, col_from:col_to + 1] = 0

            # Add this wall to the memory
            horizontal_walls.append([horizontal_wall[0], horizontal_wall[-1]])
            print("added horizontal wall ", horizontal_walls[-1])
        else:
            # replace every tile in the chosen wall by 0, so it will not be added again
            col_index = vertical_wall[0][1]
            row_from, row_to = vertical_wall[0][0], vertical_wall[-1][0]
            maze_array_np[row_from:row_to + 1, col_index] = 0

            # Add this wall to the memory
            vertical_walls.append([vertical_wall[0], vertical_wall[-1]])
            print("added vertical wall ", vertical_walls[-1])
    return horizontal_walls, vertical_walls


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


def generate_xml(map_name: str, robot_name: str, scale: float = 1., walls_height: float = 1., nb_markers: int = 0) -> \
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

    walls_width = str(scale)
    walls_height = str(walls_height)

    # Load base ant-maze file etree:

    # Setup paths
    worlds_directory_path = pathlib.Path(__file__).parent / "worlds"
    maps_directory_path = pathlib.Path(__file__).parent / "maps"
    robots_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "robots"
                           / (robot_name + ".xml"))
    block_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "block.xml")
    sphere_file_path = str(pathlib.Path(__file__).parent / "worlds" / "building_assets" / "sphere.xml")
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
            import_path = "environments.my_robotic_room.simulations_assets.maps." + map_name
        maze_array = np.array(importlib.import_module(import_path).maze_array)
    except ModuleNotFoundError:
        raise FileNotFoundError("Could not find map with name " + map_name + ".py, Please verify it is located in "
                                + str(maps_directory_path) + ".")

    # Load walls
    maze_array_shape = np.array(maze_array).shape
    horizontal_walls, vertical_walls = load_walls(maze_array)

    def add_wall(name, position, size):
        block_node = etree.parse(block_file_path).getroot()
        block_node.attrib["name"] = name

        # Set position
        block_node.find("pose").text = position + " 0 0 0"
        # Set size
        block_node.find("link").find("collision").find("geometry").find("box").find("size").text = size
        block_node.find("link").find("visual").find("geometry").find("box").find("size").text = size

        world_node.append(block_node)

    # Build walls xml nodes using loaded walls data
    for wall_id, wall in enumerate(vertical_walls):
        position_x, position_y = maze_pos_to_simulation_pos((wall[1][0] + wall[0][0]) / 2,
                                                            (wall[1][1] + wall[0][1]) / 2,
                                                            scale, maze_array)
        start_i = - (wall[1][0] - maze_array_shape[0] / 2 + 0.5)
        end_i = - (wall[0][0] - maze_array_shape[0] / 2 + 0.5)
        wall_size = str((end_i - start_i + 1) * scale) + " " + walls_width + " " + walls_height

        add_wall(name="vertical_wall_" + str(wall_id), position=str(position_x) + " " + str(position_y) + " 0",
                 size=wall_size)

    for wall_id, wall in enumerate(horizontal_walls):
        position_x, position_y = maze_pos_to_simulation_pos((wall[1][0] + wall[0][0]) / 2,
                                                            (wall[1][1] + wall[0][1]) / 2,
                                                            scale, maze_array)

        start_j = wall[0][1] - maze_array_shape[1] / 2 + 0.5
        end_j = wall[1][1] - maze_array_shape[1] / 2 + 0.5
        wall_size = walls_width + " " + str((end_j - start_j + 1) * scale) + " " + walls_height

        add_wall(name="horizontal_wall_" + str(wall_id), position=str(position_x) + " " + str(position_y) + " 0",
                 size=wall_size)

    # Load robot xml
    robot_node = etree.parse(robots_file_path).getroot()
    # Get default robot position
    start_coordinates = np.argwhere(maze_array == 2)[0]
    robot_start_position = maze_pos_to_simulation_pos(*start_coordinates, scale, maze_array)
    # Set robot position
    pose = robot_node.find("pose")
    pose_list = pose.text.split(" ")
    pose_list[0] = str(robot_start_position[0])
    pose_list[1] = str(robot_start_position[1])
    pose_list[-1] = str(-0.5*math.pi)
    pose.text = " ".join(pose_list)
    # Spawn robot
    world_node.append(robot_node)

    # Add goal_marker
    sphere_node = etree.parse(sphere_file_path).getroot()
    sphere_node.attrib["name"] = "goal_marker"
    sphere_node.find("link").find("visual").find("material").find("ambient").text = "1 0 0 1"
    world_node.append(sphere_node)

    # Generate output file from etree informations
    xml_output_path = worlds_directory_path / ("generated_" + map_name + ".world")
    tree.write(str(xml_output_path))

    # Make generated file more pretty
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_output_path, parser)
    tree.write(xml_output_path, pretty_print=True)

    return maze_array, xml_output_path


if __name__ == "__main__":
    from environments.robotic_environment.simulations_assets.maps.four_rooms import maze_array

    # maps = ["empty_room", "extreme_maze", "four_rooms", "hard_maze", "medium_maze"]
    # for map_name in maps:
    #     generate_xml(map_name)
    map_name = "four_rooms"
    robot_name = "irobot"
    generate_xml(map_name, robot_name, scale=0.5)
