cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deeplab)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(deeplab deeplab.cpp)
target_link_libraries(deeplab "${TORCH_LIBRARIES}" ${OpenCV_LIBS} stdc++fs)
set_property(TARGET deeplab PROPERTY CXX_STANDARD 14)