import os
import pprint
import shlex
import subprocess


def source(file_path):

    command = shlex.split("bash -c 'source init_env && env'")
    proc = subprocess.Popen(command, stdout = subprocess.PIPE)
    for line in proc.stdout:
      (key, _, value) = line.partition("=")
      os.environ[key] = value
    proc.communicate()

    pprint.pprint(dict(os.environ))


if __name__ == "__main__":
    source("/opt/ros/humble/setup.bash")
    source("/usr/share/gazebo/setup.sh")
    source(pathlib.Path(__file__).parent / "colcon_ws" / "install" / "setup.bash")


