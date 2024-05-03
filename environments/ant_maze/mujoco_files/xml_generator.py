import importlib
import os.path

import numpy as np
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


def generate_xml(map_name: str) -> (dict, str):
    """
    Generate an ant-maze environment model from a base model and maze walls description.
    :returns: (as a tuple of two elements)
     - A 'dict' that contains maze description file information.
     - A 'str' that correspond to the path where the resultant xml file has been generated.
    """

    # Load base ant-maze file etree:

    # Get the path to the current directory.
    current_directory = os.path.dirname(__file__)
    tree = etree.parse(current_directory + "/ant_maze.xml")

    # Find 'world_body' int ant-maze xml file:
    world_body_node = None
    for child in tree.getroot():
        if child.tag == "worldbody":
            world_body_node = child

    # Load maps data
    maze_array = np.array(importlib.import_module("sciborg.environments.maps." + map_name).maze_array)

    # Load walls
    width = len(maze_array[0])
    height = len(maze_array)
    horizontal_walls, vertical_walls = load_walls(maze_array)

    # Build walls xml nodes using loaded walls data
    for wall_id, wall in enumerate(horizontal_walls):
        start_x = wall[0][0] - width / 2 + 0.5
        end_x = wall[1][0] - width / 2 + 0.5
        position_x = (end_x + start_x) / 2
        position_y = - (wall[0][1] - height / 2 + 0.5)
        position = str(position_x) + " " + str(position_y) + " 0.5"
        wall_size = str((end_x - start_x + 1) / 2) + " 0.5 0.5"
        node = E.body(
            E.geom(type="box", size=wall_size, contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1"),
            name="horizontal_wall_" + str(wall_id), pos=position
        )
        world_body_node.append(node)

    for wall_id, wall in enumerate(vertical_walls):
        start_y = - (wall[1][1] - height / 2 + 0.5)
        end_y = - (wall[0][1] - height / 2 + 0.5)
        position_x = wall[0][0] - width / 2 + 0.5
        position_y = (end_y + start_y) / 2
        position = str(position_x) + " " + str(position_y) + " 0.5"
        wall_size = "0.5 " + str((end_y - start_y + 1) / 2) + " 0.5"
        node = E.body(
            E.geom(type="box", size=wall_size, contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1"),
            name="vertical_wall_" + str(wall_id), pos=position
        )
        world_body_node.append(node)

    xml_output_path = current_directory + "/generated/" + "ant_maze_" + map_name + ".xml"
    tree.write(xml_output_path)

    # Make generated file more pretty
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_output_path, parser)
    tree.write(xml_output_path, pretty_print=True)

    return maze_array, xml_output_path


if __name__ == "__main__":
    # maps = ["empty_room", "extreme_maze", "four_rooms", "hard_maze", "medium_maze"]
    # for map_name in maps:
    #     generate_xml(map_name)
    map_name = "four_rooms_small"
    generate_xml(map_name)
