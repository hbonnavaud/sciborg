# isae_simulations_package

This project contains gazebo simulations environment for use with ros2.

How use it:

```bash
#create a ros2 workspace with this repo:
mkdir -p ~/colcon_ws/src
cd colcon_ws/src
git clone https://gitlab.isae-supaero.fr/p.chauvin/isae_simulations_package.git
cd ..
colcon build
```

```bash
#source gazebo env and workspace env before launch:
. /usr/share/gazebo/setup.bash
. install/setup.bash
ros2 launch isae_simulations_package load_isae_world_into_gazebo.launch.py
```
