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
        first_tile = np.flip(np.argwhere(maze_array_np == 1)[0].copy()).tolist()

        # build the horizontal wall
        horizontal_wall = [first_tile]
        current_x_coordinates = first_tile[0]  # Add wall parts at the left
        while current_x_coordinates - 1 > 0 and maze_array_np[first_tile[1]][current_x_coordinates - 1].item() == 1:
            horizontal_wall.insert(0, [current_x_coordinates - 1, first_tile[1]])
            current_x_coordinates -= 1
        current_x_coordinates = first_tile[0]  # Add wall parts at the right
        while current_x_coordinates + 1 < len(maze_array[0]) \
                and maze_array_np[first_tile[1]][current_x_coordinates + 1].item() == 1:
            horizontal_wall.append([current_x_coordinates + 1, first_tile[1]])
            current_x_coordinates += 1

        # build the vertical wall
        vertical_wall = [first_tile]
        current_y_coordinates = first_tile[1]  # Add wall parts at the left
        while current_y_coordinates - 1 > 0 and maze_array_np[current_y_coordinates - 1][first_tile[0]].item() == 1:
            horizontal_wall.insert(0, [first_tile[1], current_y_coordinates - 1])
            current_y_coordinates -= 1
        current_y_coordinates = first_tile[1]  # Add wall parts at the right
        while current_y_coordinates + 1 < len(maze_array) \
                and maze_array_np[current_y_coordinates + 1][first_tile[0]].item() == 1:
            vertical_wall.append([first_tile[0], current_y_coordinates + 1])
            current_y_coordinates += 1

        # Choose the longer wall and add it to walls list.
        if len(horizontal_wall) > len(vertical_wall):
            # replace every tile in the chosen wall by 0, so it will not be added again
            row_index = horizontal_wall[0][1]
            col_from, col_to = horizontal_wall[0][0], horizontal_wall[-1][0]
            maze_array_np[row_index, col_from:col_to + 1] = 0

            # Add this wall to the memory
            horizontal_walls.append([horizontal_wall[0], horizontal_wall[-1]])
        else:
            # replace every tile in the chosen wall by 0, so it will not be added again
            col_index = vertical_wall[0][0]
            row_from, row_to = vertical_wall[0][1], vertical_wall[-1][1]
            maze_array_np[row_from:row_to + 1, col_index] = 0

            # Add this wall to the memory
            vertical_walls.append([vertical_wall[0], vertical_wall[-1]])
    return horizontal_walls, vertical_walls


def position_from_coordinates(x, y, scale, maze_array):
    width, height = np.array(maze_array).shape
    return (((width - 1 - x) - width / 2 + 0.5) * scale,
            ((height - 1 - y) - height / 2 + 0.5) * scale)


def generate_xml(map_name: str, robot_name: str, scale: float = 1., walls_height: float = 1.) -> (dict, str):
    """
    Generate an ant-maze environment model from a base model and maze walls description.
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
    width, height = np.array(maze_array).shape
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
    for wall_id, wall in enumerate(horizontal_walls):
        position_x, position_y = position_from_coordinates((wall[1][0] + wall[0][0]) / 2,
                                                           (wall[1][1] + wall[0][1]) / 2,
                                                           scale, maze_array)

        start_x = wall[0][0] - width / 2 + 0.5
        end_x = wall[1][0] - width / 2 + 0.5
        wall_size = str((end_x - start_x + 1) * scale) + " " + walls_width + " " + walls_height

        add_wall(name="horizontal_wall_" + str(wall_id), position=str(position_y) + " " + str(position_x) + " 0",
                 size=wall_size)

    for wall_id, wall in enumerate(vertical_walls):
        position_x, position_y = position_from_coordinates((wall[1][0] + wall[0][0]) / 2,
                                                           (wall[1][1] + wall[0][1]) / 2,
                                                           scale, maze_array)
        start_y = - (wall[1][1] - height / 2 + 0.5)
        end_y = - (wall[0][1] - height / 2 + 0.5)
        wall_size = walls_width + " " + str((end_y - start_y + 1) * scale) + " " + walls_height

        add_wall(name="vertical_wall_" + str(wall_id), position=str(position_y) + " " + str(position_x) + " 0",
                 size=wall_size)

    # Load robot xml
    robot_node = etree.parse(robots_file_path).getroot()
    # Get default robot position
    start_coordinates = np.argwhere(maze_array == 2)[0]
    robot_start_position = position_from_coordinates(*start_coordinates, scale, maze_array)
    # Set robot position
    pose = robot_node.find("pose")
    pose_list = pose.text.split(" ")
    pose_list[0] = str(robot_start_position[1])
    pose_list[1] = str(robot_start_position[0])
    pose_list[-1] = str(math.pi)
    pose.text = " ".join(pose_list)
    # Spawn robot
    world_node.append(robot_node)

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
