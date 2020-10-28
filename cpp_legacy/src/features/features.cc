
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <algorithm>
#include "./../ara_slam.h"

namespace cv_xft = cv::xfeatures2d;

namespace ara_slam { namespace features {
    using features_t = std::tuple<std::vector<cv::KeyPoint>,cv::Mat>;    

    std::vector<cv::DMatch> compare_features(cv::Mat d1, cv::Mat d2, feature_matcher fm){
        std::vector<cv::DMatch> matches;
        double min_dist, max_dist;
        if( fm == feature_matcher::brute ){
            const float kGoodPercentMatch = 0.25f;            
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(d1,d2,matches,cv::Mat());
            std::sort(matches.begin(), matches.end());
            //filter matches
            const int kNumMatches = matches.size() * kGoodPercentMatch;
            matches.erase(matches.begin() + kNumMatches, matches.end());

        }else if( fm == feature_matcher::flann ){
            cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(5, 10, 2));
            matcher.match(d1, d2, matches);
            //find the minimum and maximum distance
            std::for_each(matches.begin(), matches.end(), [min_dist, max_dist](const auto m) mutable{
                min_dist = (m.distance < min_dist) ? m.distance : min_dist;
                if( m.distance > max_dist ){ max_dist = m.distance; }
            });
            double dist_thresh = std::max(min_dist * 2.0, 30.0);
            //filter the matches based on the distance threshold
            auto ret = std::remove_if(matches.begin(), matches.end(), [dist_thresh](const auto m){
                return m.distance > dist_thresh; });
            matches.erase(ret, matches.end()); //resize the container as `remove_if` does not do it
        }

        //remove duplicate matches; d1 has queryIdx and d2 has trainIdx
        //so remove the many-to-one relationship in d2
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2){
            return m1.trainIdx < m2.trainIdx;
        });
        auto ret = std::unique(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) -> bool {
            return m1.trainIdx == m2.trainIdx;
        });
        matches.erase(ret, matches.end());

        return matches;
    }

    features_t extract_features_orb(cv::Mat image){
        // const int kMaxFeatures = 250;        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;        
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create();        
        orb->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        return std::make_tuple(keypoints, descriptors);
    }

    features_t extract_features_sift(cv::Mat image){      
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;        
        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();        
        sift->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        return std::make_tuple(keypoints, descriptors);
    }

    features_t extract_features_fast(cv::Mat image){
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(image, keypoints, 25);
        // std::sort(keypoints.begin(), keypoints.end(), [](auto p1, auto p2){
        //     return (p1.pt.x < p2.pt.x) && (p1.pt.y < p2.pt.y);
        // });
        return std::make_tuple(keypoints, cv::Mat());
    }

    auto extract_features(cv::Mat image, feature_extractor fe) -> features_t {
        cv::Mat _image;        
        if( image.channels() == 3){
            cv::cvtColor(image, _image, cv::COLOR_BGR2GRAY);
        }else{
            _image = image;
        }

        switch (fe){
            case feature_extractor::orb:
                return extract_features_orb(_image);
            case feature_extractor::fast:
                return extract_features_fast(_image);
            case feature_extractor::sift:
                return extract_features_sift(_image);                
            default:
                return extract_features_orb(_image);
        }
    }
}}
