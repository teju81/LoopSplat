#!/bin/bash

colcon build
source install/setup.bash
ros2 run loopsplat_ros loopclosure_detection_test