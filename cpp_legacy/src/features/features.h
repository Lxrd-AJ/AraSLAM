#ifndef _DESCRIPTOR_
#define _DESCRIPTOR_

#include <opencv2/opencv.hpp>
#include "./../ara_slam.h"

namespace ara_slam { namespace features {

    using descriptors_t = cv::Mat;
    using features_t = std::tuple<std::vector<cv::KeyPoint>,descriptors_t>;

    // std::vector<cv::Point> points_from_matches(std::vector<cv::KeyPoint>, std::vector<cv::DMatch>);

    std::vector<cv::DMatch> compare_features(cv::Mat d1, cv::Mat d2, feature_matcher fm = feature_matcher::brute);    

    auto extract_features(cv::Mat image, feature_extractor fe = feature_extractor::orb) -> features_t;
}}


#endif