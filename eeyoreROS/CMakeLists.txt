cmake_minimum_required(VERSION 3.0.2)
project(eeyore_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../eeyore/cmake")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  image_transport
  cv_bridge
  eeyore
)

find_package(OpenCV 4 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES eeyore_ros
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp sensor_msgs std_msgs eeyore
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include/boson
)

#target_link_libraries(${PROJECT_NAME}
#  ${OpenCV_INCLUDE_DIRS}
#  ${catkin_LIBRARIES}
#)


add_executable(boson_node
  src/boson_wrapper.cpp
  src/boson_node.cpp
)

add_dependencies(boson_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(boson_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
