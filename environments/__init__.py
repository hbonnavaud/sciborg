from .environment import Environment
from .goal_conditioned_environment import GoalConditionedEnvironment
import importlib
from .. import settings

from .robotic_environment_v2 import *
from .robotic_environment_v3 import *
from .robotic_environment_v4 import *

# For every environment module, we use try except statements in case one environment have unmeet dependencies but the
# user wants to use the others.
# For an example, it allows you to import GridWorld without having mujoco or vizdoom installed.

# Import grid-world
try:
    from .grid_world import *
except Exception as e:
    if not settings.supress_import_warnings:
        print(f"Warning: module 'grid_world' cannot be imported due to the following error: ", e, sep="")

# Import point-maze
try:
    from .point_maze import *
except Exception as e:
    if not settings.supress_import_warnings:
        print(f"Warning: module 'point_maze' cannot be imported due to the following error: ", e, sep="")

# Import ant-maze
try:
    from .ant_maze import *
except Exception as e:
    if not settings.supress_import_warnings:
        print(f"Warning: module 'ant_maze' cannot be imported due to the following error: ", e, sep="")

# Import doom-maze
try:
    from .doom_maze import *
except Exception as e:
    if not settings.supress_import_warnings:
        print(f"Warning: module 'doom_maze' cannot be imported due to the following error: ", e, sep="")

# Import robotic environment
try:
    from .robotic_environment import *
except Exception as e:
    if not settings.supress_import_warnings:
        print(f"Warning: module 'robotic_environment' cannot be imported due to the following error: ", e, sep="")
