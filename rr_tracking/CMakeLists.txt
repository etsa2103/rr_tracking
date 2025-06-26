cmake_minimum_required(VERSION 3.0.2)
project(rr_tracking)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/blaze_pose_node.py
  scripts/clahe_filter_node.py
  scripts/display_node.py
  scripts/rr_tracking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
