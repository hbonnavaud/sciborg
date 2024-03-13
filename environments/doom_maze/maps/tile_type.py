from enum import Enum


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    TERMINAL = 3