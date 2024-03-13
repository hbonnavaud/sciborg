try:
    from .send_discord_message import send_discord_message, EmptyWebhookLinkError
except ModuleNotFoundError as e:
    module_name = e.name if e.name != "omg" else "omgifol"
    # print("\n  > WARNING, cannot use send_discord_message function because of missing module '" + module_name + "'")
except Exception as e:
    # print("\n  > WARNING, cannot use send_discord_message function because of unknown error.")
    raise e

from .sys_fun import create_dir, empty_dir, save_image, generate_video
from .stopwatch import Stopwatch
from .general import (get_euler_from_quaternion, get_quaternion_from_euler, get_dict_as_str,
                      get_point_image_after_rotation, print_replace_above)
from .progress_bar import ProgressBar, Loader
from .drawing import *
from .spaces import *
from .ansi import Ansi
