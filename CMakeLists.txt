cmake_minimum_required(VERSION 3.15)
project(TraitementImages LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(ti_opencv INTERFACE)
target_include_directories(ti_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ti_opencv INTERFACE ${OpenCV_LIBS})

add_library(color_distribution STATIC
  src/color_distribution/color_distribution.cpp
  src/color_distribution/color_distribution.h
)
target_include_directories(color_distribution PUBLIC ${PROJECT_SOURCE_DIR}/src/color_distribution)
target_link_libraries(color_distribution PRIVATE ti_opencv)

add_library(callbacks STATIC
  src/callbacks/reconnaissance.cpp
  src/callbacks/reconnaissance.h
  src/callbacks/tests.cpp
  src/callbacks/tests.h
)
target_link_libraries(callbacks PRIVATE ti_opencv color_distribution)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ti_opencv color_distribution callbacks)
