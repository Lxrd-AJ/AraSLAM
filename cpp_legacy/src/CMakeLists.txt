include(CMakeToolsHelpers OPTIONAL)
link_directories(${PCL_LIBRARY_DIRS})

SET ( SOURCES
    calibration/calibrator.cc
    epipolar_geometry/epipolar.cc
    features/features.cc
    depth/depth_map.cc
    viz/viz.cc
    localisation/localisation.cc
    data/dataset.cc
    data/kitti_dataset.cc
    data/video_dataset.cc
    visual_odometry.cc
)

SET ( HEADERS
    ara_slam.h
    calibration/calibrator.h
    epipolar_geometry/epipolar.h
    features/features.h
    depth/depth_map.h
    viz/viz.h
    localisation/localisation.h
    data/dataset.h
    data/kitti_dataset.h
    data/video_dataset.h
    visual_odometry.h
)

SET ( ARA_SLAM_INCLUDE_FOLDERS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

set( ARA_SLAM_LINK_LIBRARIES 
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

include_directories(ARA_SLAM_INCLUDE_FOLDERS )
add_definitions(${PCL_DEFINITIONS})


add_library( Ara_SLAM STATIC ${SOURCES} ${HEADERS} )

target_link_libraries( Ara_SLAM ${ARA_SLAM_LINK_LIBRARIES})

install( TARGETS Ara_SLAM
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
)

install( DIRECTORY . DESTINATION include/ara_slam
    FILES_MATCHING PATTERN "*.h"
)