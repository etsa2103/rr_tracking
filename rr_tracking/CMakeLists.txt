cmake_minimum_required(VERSION 3.0.2)
project(rr_tracking)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  sensor_msgs
  message_generation
  rospy
)

## Declare messages
add_message_files(
  FILES
  StableRange.msg
  ProcessingState.msg
  TrackingState.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime
)

## Install your Python nodes
catkin_install_python(PROGRAMS
  scripts/facial_tracking_node.py
  scripts/gui_node.py
  scripts/signal_processing_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
