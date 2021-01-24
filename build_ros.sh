#!/bin/sh
echo "Building ROS nodes"

workspace=$(cd $(dirname $0); pwd)
echo $workspace
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:${workspace}/Examples/ROS
cd Examples/ROS/ORB_SLAM2
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
