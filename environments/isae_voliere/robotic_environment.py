import pathlib
import os
import pprint
import shlex
import subprocess
import roslaunch
import rospy


def source(file_path):
    command = shlex.split("bash -c 'source " + file_path + " && env'")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    for line in proc.stdout:
        (key, _, value) = line.partition("=")
        os.environ[key] = value
    proc.communicate()

    pprint.pprint(dict(os.environ))


class RoboticEnvironment:

    def __init__(self):

        ###############################################################################
        #############               LAUNCH GAZEBO SIMULATOR               #############
        ###############################################################################

        # Source necessary files
        source("/opt/ros/humble/setup.bash")  # Source ros humble
        source("/usr/share/gazebo/setup.sh")  # Source gazebo
        source(pathlib.Path(__file__).parent / "colcon_ws" / "install" / "setup.bash")  # Source our env sources.

        # Launch the simulation in a sub-process


        rospy.init_node('tester', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [pathlib.Path(__file__).parent
                                                         / "colcon_ws/install/isae_simulations_package/share/isae_simulations_package/launch/load_isae_world_into_gazebo.launch.py"])

        launch.start()

        launch.shutdown()
        # subprocess.Popen("ros2", "launch", "isae_simulations_package", "load_isae_world_into_gazebo.launch.py")


if __name__ == "__main__":
    RoboticEnvironment()