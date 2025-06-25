
# Boson Thermal Camera ROS Workspace

This repository contains a complete ROS Noetic workspace for working with the Boson 640+ thermal camera to monitor resperation rate (rr).

## Workspace Structure
```
catkin_ws/
├── blaze_pose/          # Python package for pose tracking that is used to track nose and mouth
├── boson_camera/        # C++ driver and ROS node for interfacing with the Boson camera
├── boson_utils/         # Custom python package used for filtering, recording, and playback with boson
├── rr_tracking/         # Custom scripts for respiration rate tracking
├── catkin_simple/       # Catkin Simple build system (was needed for boson_camera)
├── CMakeLists.txt       # Top-level symlink to ROS toplevel.cmake
└── readme.md            # You are here
```

## Launch Files
| Launch File | Description |
|-------------|-------------|
| `boson_camera/launch/boson_live.launch` | Starts the Boson camera driver to broadcast the video on /boson/image_raw and launches RViz|
| `boson_utils/launch/loop_boson.launch` | Replays specified bag file in a loop and launches RViz |
| `blaze_pose/launch/track_breath_centroid.launch` | Runs BlazePose node on incoming thermal image stream |

## Setup Notes

Ensure you’ve built the workspace and sourced it:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Packages Overview

### `boson_camera`
- Nodes:
  - `boson_ros_node.cpp`
    - Publishes `mono16` audio on `/boson/image_raw`.
- Launch options:  
    - To stream live video feed
        ```bash
        roslaunch boson_utils/boson640.launch
        ```
    - To stream live video feed and open a display
        ```bash
        roslaunch boson_camera boson_live.launch
        ```

### `boson_utils`
- Python utilities for filtering, recording, and playback.
- Nodes:
  - `clahe_filter_node.py`: Optional histogram equalization filter.
- Launch option:
  - To record and bag video
    ```bash
    roslaunch boson_camera boson_live.launch
    rosbag record /boson/image_raw
    ```
  - To loop bagged video and open a display
    ```bash
    roslaunch boson_utils loop_boson.launch
    ```
- Notes:
    - ADD NOTE ABOUT WHERE BAG IS SAVED TO AND HOW TO CHOOSE WHICH BAG TO LOOP

### `blaze_pose`
- Python node using BlazePose for mouth and nose tracking

- Nodes:
  - `blazePose_node.py`
    - Subscribes to `/boson/image_raw`,
    - Publishes BLANK on `/boson/breath_centroid`.
- Launch option:
  - To run face tracking
    ```bash
    roslaunch blaze_pose track_breath_centroid.launch
    ```

### `rr_tracking`
- Placeholder for respiration rate tracking based on breath centroid or temperature data.

## TODO

- [x] Record thermal images
- [x] Allow bagging and replaying in rviz
- [x] Track mouth and nose with BlazePose
- [ ] Respiration rate estimation

---