import os
import random

import numpy as np
import importlib
from omg import *

def get_position(w, h, block_size):
    x, y = w * block_size, h * block_size
    x += int(block_size / 2)
    y += int(block_size / 2)
    return x, y


def build_wall(map_array, random_walls_textures=False, block_size=96, discrete_goals_set=False):
    things = []
    linedefs = []
    vertexes = []
    v_indexes = {}

    max_w = len(map_array[0]) - 1
    max_h = len(map_array) - 1

    def __is_edge(w, h):
        return w in (0, max_w) or h in (0, max_h)

    def __add_start(w, h):
        x, y = get_position(w, max_h - h, block_size)
        things.append(ZThing(*[len(things) + 1000, x, y, 0, 0, 9001, 22279]))

    def __add_vertex(w, h):
        if (w, h) in v_indexes:
            return
        x, y = get_position(w, max_h - h, block_size)
        print("adding vertex at ", [x, y])
        v_indexes[w, h] = len(vertexes)
        vertexes.append(Vertex(x, y))

    def __add_line(start, end, edge=False):
        assert start in v_indexes
        assert end in v_indexes

        mask = 1
        if random_walls_textures:
            left = random.randint(0, 66)
            right = random.randint(0, 66)
        else:
            left = right = 0
        if __is_edge(*start) and __is_edge(*end):
            if not edge:
                return
            else:
                # Changed the right side (one towards outside the map)
                # to be -1 (65535 for Doom)
                right = 65535
                mask = 15

        # Flipped end and start vertices to make lines "point" at right direction (mostly to see if it works)
        line_properties = [v_indexes[end], v_indexes[start], mask
                           ] + [0] * 6 + [left, right]
        line = ZLinedef(*line_properties)
        linedefs.append(line)

    for h, row in enumerate(map_array):
        for w, block in enumerate(row):
            if block == 1:
                __add_vertex(w, h)
            else:
                pass

    corners = [(0, 0), (max_w, 0), (max_w, max_h), (0, max_h)]

    for i in range(len(corners)):
        if i != len(corners) - 1:
            p_1 = corners[i + 1]
            p_2 = corners[i]
        else:
            p_1 = corners[0]
            p_2 = corners[i]

        if p_1[1] > p_2[1]:
            for h in reversed(range(p_2[1], p_1[1])):
                __add_line((p_1[0], h + 1), (p_2[0], h), True)
        elif p_1[1] < p_2[1]:
            for h in range(p_1[1], p_2[1]):
                __add_line((p_1[0], h), (p_2[0], h + 1), True)
        elif p_1[0] > p_2[0]:
            for w in reversed(range(p_2[0], p_1[0])):
                __add_line((w + 1, p_1[1]), (w, p_2[1]), True)
        elif p_1[0] < p_2[0]:
            for w in range(p_1[0], p_2[0]):
                __add_line((w, p_1[1]), (w + 1, p_2[1]), True)

    # Now connect the walls
    for h, row in enumerate(map_array):
        for w, _ in enumerate(row):
            if (w, h) not in v_indexes:
                __add_start(w, h)
                continue

            if (w + 1, h) in v_indexes:
                __add_line((w, h), (w + 1, h))

            if (w, h + 1) in v_indexes:
                __add_line((w, h), (w, h + 1))

    # Place goals items
    if discrete_goals_set:
        goals_positions = np.where(map_array == 3)
        goals_types = [32, 35, 30, 41, 36, 70, 56, 55, 47, 44]
        for goal_id in range(len(goals_positions[0])):
            h = goals_positions[0][goal_id]
            w = goals_positions[1][goal_id]
            goal_type = goals_types[goal_id % len(goals_types)]

            x, y = get_position(w, max_h - h, block_size)
            things.append(ZThing(0, x, y, 0, 0, goal_type, 263))
    else:
        candidates = np.where(map_array == 2)
        position_index = random.randint(0, len(candidates[0]) - 1)
        h = 1
        w = 3
        x, y = get_position(w, max_h - h, block_size)
        things.append(ZThing(0, x, y, 0, 0, 32, 263))  # type 32 = red pillar

    return things, vertexes, linedefs

def build_wad(output_directory: str, map_array,
              random_walls_textures=True,
              discrete_goals_set=False,
              behaviour_file="static_goal.acs", block_size=96):

    if not output_directory.endswith("/"): output_directory += "/"
    assert os.path.isdir(output_directory)

    # Verify inputs
    ## behaviour_file
    configurations_files_path = os.path.dirname(os.path.abspath(__file__)) + "/configurations/"
    if not "." in behaviour_file:
        behaviour_file += ".acs"
    if behaviour_file.split(".")[-1] != "acs":
        raise ValueError("Wrong behaviour file extension '." + behaviour_file.split(".")[-1] + "'")
    behaviour_file_path = configurations_files_path + behaviour_file
    if not os.path.exists(behaviour_file_path):
        raise ValueError("File " + behaviour_file + "not found in directory " + configurations_files_path)

    ## map_array
    assert isinstance(map_array, np.ndarray)

    # Build wad file
    new_wad = WAD()
    new_map = MapEditor()
    new_map.Linedef = ZLinedef
    new_map.Thing = ZThing
    new_map.behavior = Lump(from_file=behaviour_file_path or None)
    new_map.scripts = Lump(from_file=None)

    new_map.sectors = [Sector(0, 128, 'CEIL5_2', 'CEIL5_2', 240, 0, 0)]
    # new_map.sectors = [Sector(0, 128, 'ZIMMER8', 'ZZWOLF1', 240, 0, 0)]
    things, vertexes, linedefs = build_wall(map_array,
                                            random_walls_textures=random_walls_textures,
                                            discrete_goals_set=discrete_goals_set,
                                            block_size=block_size)

    candidates = np.where(map_array == 2)
    cand_index = random.randint(0, len(candidates[0]) - 1)
    w = candidates[0][cand_index]
    h = candidates[0][cand_index]
    x, y = get_position(w, len(map_array) - h - 1, block_size)
    new_map.things = things + [ZThing(0, x, y, 0, 0, 1, 7)]

    new_map.vertexes = vertexes
    new_map.linedefs = linedefs

    texture_names = ['ZZWOLF6', 'ZIMMER8', 'TEKBRON1', 'WOOD5', 'WOODMET3', 'ZDOORB1', 'ZZWOLF9', 'TEKGREN1',
    'WOOD6', 'WOODMET4', 'TEKLITE', 'TEKWALL1', 'WOODMET2', 'TANROCK4', 'ZZWOLF10', 'WOOD9', 'ZZZFACE6', 'TANROCK3',
    'ZZZFACE2', 'ZZZFACE1', 'ZZWOLF11', 'TEKGREN4', 'ZZWOLF4', 'TEKBRON2', 'TEKGREN3', 'WOOD3', 'ZZZFACE4', 'WOOD1',
    'ZDOORF1', 'ZIMMER4', 'ZZZFACE5', 'TEKWALL6', 'WOOD4', 'TANROCK2', 'TEKGREN2', 'ZIMMER7', 'TANROCK8', 'ZZWOLF7',
    'ZZZFACE8', 'ZIMMER5', 'WOODSKUL', 'ZZWOLF3', 'ZZWOLF2', 'ZZZFACE9', 'TEKWALL2', 'ZELDOOR', 'ZIMMER2', 'WOOD8',
    'TEKWALL5', 'TANROCK5', 'TEKWALL4', 'TEKLITE2', 'ZZWOLF12', 'WOODMET1', 'ZZZFACE3', 'ZIMMER3', 'ZZWOLF1',
    'TANROCK7', 'WOODGARG', 'ZZWOLF13', 'WOOD12', 'ZZZFACE7', 'TEKGREN5', 'WOODVERT', 'ZZWOLF5', 'WOOD7',
    'ZIMMER1']

    new_map.sidedefs = [Sidedef(0, 0, '-', '-', texture_name, 0) for texture_name in texture_names]
    new_map.sidedefs.append(Sidedef(0, 0, '-', '-', '-', 0))

    new_wad.maps['MAP00'] = new_map.to_lumps()
    new_wad.to_file(output_directory + "/map" + ".wad")
