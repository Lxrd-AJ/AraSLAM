include(FetchContent)

set( ARA_SLAM_INCLUDE_FOLDERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenCV_INCLUDE_DIR}    
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.8.1
)

FetchContent_GetProperties(googletest)
if( NOT googletest_POPULATED )
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

include_directories( ${ARA_SLAM_INCLUDE_FOLDERS} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable( araslam_tests 
    epipolar/test_epipolar.cc
)

target_link_libraries( araslam_tests AraSLAM gtest_main)

add_test(NAME unit COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_DIR}/araslam_tests)