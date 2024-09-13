#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# Set the environment variable for Fast DDS
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

# Source the ROS 2 Humble setup script
source /opt/ros/humble/setup.bash

cd IsaacSim-ros_workspaces/${ROS_DISTRO}_ws
source install/setup.bash
cd ../../go2_omniverse_ws
source install/setup.bash
cd ..
if ip link show eno1 | grep -q "state UP"; then
    source ~/Desktop/go2_rl_ws/go2_rl_ws/src/unitree_ros2/setup.sh
else
    echo "Network interface eno1 is not found or not UP. Skipping setup."
fi

# Run the Python script
python omniverse_sim.py #--livestream 2