from .environment import Environment
from .goal_conditioned_environment import GoalConditionedEnvironment
from .grid_world import *
from .point_maze import *
from .robotic_environment import *

failed_one_import = False


def print_warning(message):
    global failed_one_import
    if not failed_one_import:
        print("\n")
        failed_one_import = True
    print(message)


# Import vizdoom_maze if available
try:
    from .doom_maze import *
except ModuleNotFoundError as e:
    module_name = e.name if e.name != "omg" else "omgifol"
    # print_warning("  > WARNING, cannot use vizdoom_maze because of missing module '" + module_name + "'")
except Exception as e:
    # print_warning("  > WARNING, cannot use vizdoom_maze because of unknown error.")
    raise e

# Import ant_maze if available
try:
    from .ant_maze import *
except ModuleNotFoundError as e:
    module_name = e.name
    # print_warning("  > WARNING, cannot use ant_maze because of missing module '" + module_name + "'")
except Exception as e:
    if e.args[0].split("\n")[-1].startswith("export LD_LIBRARY_PATH="):
        pass
        # print_warning("  > WARNING, cannot use ant_maze because of wrong environment variable value." +
        #               "\n    Add '" + e.args[0].split("\n")[-1] + "' to your bashrc and try again.")
    else:
        # print_warning("  > WARNING, cannot use ant_maze because of unknown error.")
        raise e

# if failed_one_import:
#     print("")
