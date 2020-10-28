find_package (Boost 1.68.0 REQUIRED COMPONENTS system filesystem)

set( CPP_SAMPLE_SOURCES
    calibration_tool.cc
    camera_pose_estimator.cc
    cam_model_detection.cc
    stereo_visual_odometry.cc
    kitti_mono_visual_odometry.cc
    mono_odometry.cc
)

set( ARA_SLAM_INCLUDE_FOLDERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../src
    ${OpenCV_INCLUDE_DIR}
    # ${YAML_CPP_INCLUDE_DIR}
)

set ( TOOLS_BUILD_LIBRARIES
    # ${YAML_CPP_LIBRARIES}
    # yaml-cpp
    # ${PCL_LIBRARY_DIRS}
)



# set (Boost_NO_SYSTEM_PATHS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (BOOST_ALL_DYN_LINK OFF)

if (Boost_FOUND)
    include_directories (SYSTEM ${Boost_INCLUDE_DIR})
endif ()

include_directories( ${ARA_SLAM_INCLUDE_FOLDERS} ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${PCL_LIBRARY_DIRS})

foreach( file ${CPP_SAMPLE_SOURCES} )

    get_filename_component( sample_name ${file} NAME_WE)
    add_executable( ${sample_name} ${file} )
    target_link_libraries( ${sample_name} Ara_SLAM ${Boost_LIBRARIES} ${TOOLS_BUILD_LIBRARIES}) #yaml-cpp

endforeach()