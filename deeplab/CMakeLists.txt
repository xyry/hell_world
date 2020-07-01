cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deeplab)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    )

add_executable(deeplab deeplab.cpp)
target_link_libraries(deeplab 
"${TORCH_LIBRARIES}" 
${OpenCV_LIBS} 
stdc++fs
${EIGEN3_LIBS})
set_property(TARGET deeplab PROPERTY CXX_STANDARD 14)