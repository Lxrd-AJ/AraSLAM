#ifndef _LOCALISATION_
#define _LOCALISATION_

#include <opencv2/opencv.hpp>
#include "./../ara_slam.h"

/**
 * - TODO: Maybe move the following funcs into the `epipolar` namespace
 *      - `recover_pose_essential` etc
 */
namespace ara_slam { namespace localisation {

    /**
     * @Deprecated
     * Left here for backwards compatibility
     */
    std::tuple<cv::Mat,cv::Mat> recover_pose(std::vector<cv::Point2f> p1,std::vector<cv::Point2f> p2,cv::Mat K);


    std::tuple<std::vector<cv::Point2f>,std::vector<cv::Point2f>> track_keypoints(cv::Mat first,std::vector<cv::Point2f> points,cv::Mat second);

    
    std::vector<cv::KeyPoint> filter_keypoints(std::vector<cv::KeyPoint>, std::vector<bool>);

    /**
     * Most likely going to be deprecated as this has now been moved to the depth module
     * If the key point has no corresponding depth value, it is dropped from the results
     * */
    std::vector<cv::Point3i> xyd_from_keypoints(std::vector<cv::KeyPoint>, cv::Mat);

    /**
     * the inlier detections
     * This uses the maximum clique algorithm to detect inlier
     * 
     * */
    std::vector<int> max_clique(std::vector<cv::Point3f> t_1, std::vector<cv::Point3f> t);
}}


#endif