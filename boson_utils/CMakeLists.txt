cmake_minimum_required(VERSION 3.0.2)
project(boson_utils)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/clahe_filter_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
