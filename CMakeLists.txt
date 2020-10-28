cmake_minimum_required(VERSION 3.4)

project (AraSLAM)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.8 REQUIRED)
# find_package(Pangolin 0.4 REQUIRED)

set(CMAKE_CXX_STANDARD 17)


# add_subdirectory(dependencies/yaml-cpp)
add_subdirectory(src)

add_subdirectory(tests)
#Build testing compulsory for now
#if( BUILD_TESTING )
#    add_subdirectory(tests)
#endif()

add_subdirectory(tools)
