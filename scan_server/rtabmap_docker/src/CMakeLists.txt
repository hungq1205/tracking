cmake_minimum_required(VERSION 3.10)
project(rtabmap_server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Standard pattern from RTAB-Map's own "Cplusplus RGBD Mapping" wiki tutorial
# for building an external project against an installed RTAB-Map — RTABMap's
# CMake config exports RTABMap_INCLUDE_DIRS / RTABMap_LIBRARIES.
find_package(RTABMap REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${RTABMap_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(rtabmap_server rtabmap_server.cc)

target_link_libraries(rtabmap_server
    ${RTABMap_LIBRARIES}
    ${OpenCV_LIBS}
    zmq
)
