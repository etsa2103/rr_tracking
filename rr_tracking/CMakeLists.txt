cmake_minimum_required(VERSION 3.0.2)
project(rr_tracking)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/facial_tracking_node.py
  scripts/gui_node.py
  scripts/signal_processing_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
