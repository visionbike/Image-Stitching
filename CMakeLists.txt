cmake_minimum_required(VERSION 3.8)
project(Image_Stitching)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED)

include_directories(${PROJECT_NAME} ${OpenCV_INCLUDE_DIRS} $ENV{EIGEN3_INCLUDE_DIR})

add_executable(Image_Stitching main.cpp image_stitcher.hpp image_stitcher.cpp feature.hpp feature.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen)