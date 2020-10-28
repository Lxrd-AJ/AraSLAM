#ifndef _DEPTH_MAP_
#define _DEPTH_MAP_

#include <opencv2/opencv.hpp>
#include "./../ara_slam.h"
#include "./../calibration/calibrator.h"

namespace ara_slam { namespace depth {
    namespace calib = ara_slam::calibration;

    /**
     * TODO: This function needs to be refined
     * The depth map needs to be smoothed out, see https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html 
     * and this tutorial https://walchko.github.io/blog/Vision/stereo-vision/stereo.html#Pretty-Disparity-Map
     * */
    cv::Mat compute_depth(cv::Mat, cv::Mat);
    
    /**
     * TODO: 
     * - Rename the function: triangulate_points is a bit misleading as it is more suited to 
     *      mono slam. Try renaming it to `disparity_depth_Q`
     * */
    std::vector<cv::Point3i> triangulate_points(std::vector<cv::Point3i>, calib::stereo_t);

    /**
     * If the key point has no corresponding depth value, it is dropped from the results
     * */
    std::vector<cv::Point3f> world_xyz_from_keypoints(std::vector<cv::KeyPoint>, cv::Mat);

    //TODO: Add function to convert a color image & disparity map into a 3D image https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02 we already have one for grayscale images

    /**
     * projects a 3D world point into the camera coordinates using a projection matrix
     * */
    std::vector<cv::Point2i> world_xyz_to_image(std::vector<cv::Point3f>, calib::stereo_t);
}}


#endif