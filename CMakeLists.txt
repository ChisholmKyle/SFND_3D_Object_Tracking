cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

set(OpenCV_DIR /home/kyle/.local/opt/OpenCV-4.2.0/lib/cmake/opencv4)
find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_compile_definitions(3D_object_tracking PRIVATE DATA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/")
target_compile_options(3D_object_tracking PRIVATE -Wall)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})
