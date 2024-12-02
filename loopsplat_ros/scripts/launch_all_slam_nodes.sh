#!/bin/bash

colcon build
source install/setup.bash

# Name of the tmux session
SESSION="loopsplat_ros_session"

# ROS executables (replace these with your actual ROS executables)
RUN_FRONTEND="ros2 run loopsplat_ros frontend_slam"
RUN_BACKEND="ros2 run loopsplat_ros backend_slam"
RUN_GUI="ros2 run loopsplat_ros slam_gui"

# Check if the tmux session already exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    # Step 1: Create a new tmux session and window
    tmux new-session -d -s $SESSION -n "ROS"

    # Step 2: Split the window into 4 panes (2x2 grid)
    tmux split-window -h  # Split horizontally
    tmux split-window -v  # Split the left pane vertically
    tmux select-pane -t 0  # Move focus to the first pane (top-left)
    tmux split-window -v  # Split the right pane vertically

    # Step 3: Send ROS commands to each pane
    tmux send-keys -t 0 "$RUN_FRONTEND" C-m  # Top-left pane
    tmux send-keys -t 1 "$RUN_BACKEND" C-m  # Top-right pane
    tmux send-keys -t 2 "$RUN_GUI" C-m  # Bottom-left pane
    #tmux send-keys -t 3 "$RUN_LOOPCLOSING" C-m  # Bottom-right pane

    # Step 4: Attach to the tmux session
    tmux attach-session -t $SESSION
else
    echo "Session $SESSION already exists. Attaching to it."
    tmux attach-session -t $SESSION
fi