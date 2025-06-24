cmake_minimum_required(VERSION 3.0.2)
project(blaze_pose)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/blazePose_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
