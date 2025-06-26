
# Boson Thermal Camera ROS Workspace

This repository contains a complete ROS Noetic workspace for working with the Boson 640+ thermal camera to monitor resperation rate (rr).

## Workspace Structure
```
catkin_ws/
├── boson_camera/        # C++ driver and ROS node for interfacing with the Boson camera
├── catkin_simple/       # Catkin Simple build system (was needed for boson_camera)
├── rr_tracking/         # Custom scripts for respiration rate tracking
├── CMakeLists.txt       # Top-level symlink to ROS toplevel.cmake
└── readme.md            # You are here
```

## Launch Files
| Launch File | Description |
|-------------|-------------|
| `boson_camera/launch/boson_with_display.launch` | Starts Boson camera driver to broadcast the video on /boson/image_raw and launches display|
| `boson_camera/launch/boson640.launch` | Starts Boson camera driver to broadcast the video on /boson/image_raw|
| `rr_tracking/launch/full_system_live.launch` | Has boson stream live video, runs blazePose on raw video to get pixels under the nose, analyzes these pixels to get resperation rate |
| `rr_tracking/launch/full_system_loop.launch` | Run ros bag of old boson recording, runs blazePose on raw video to get pixels under the nose, analyzes these pixels to get resperation rate |

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
- Config options:  
    - Edit `boson640_config.yaml` to config camera settings

### `Catkin_simple`
- Needed for boson_camera and used for CMakeList files
- No need to touch

### `rr_tracking`
- Nodes:
  - `blaze_pose_node.cpp`
    - Publishes `mono16` audio on `/boson/image_raw`.
  - `clahe_filter_node.cpp`
    - Publishes `mono16` audio on `/boson/image_raw`.
  - `display_node.cpp`
    - Publishes `mono16` audio on `/boson/image_raw`.
  - `rr_tracking_node.cpp`
    - Publishes `mono16` audio on `/boson/image_raw`.

## TODO
- [x] Record thermal images
- [x] Track face, then nose
- [x] Get preliminary rr results
- [ ] Improve facial tracking
- [ ] Improve signal filtering
- [ ] Test in different environments on different people

---