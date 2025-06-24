cmake_minimum_required(VERSION 3.0.2)
project(rr_tracking)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/rr_tracking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
