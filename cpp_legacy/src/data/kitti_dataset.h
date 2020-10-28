#ifndef _KITTI_DATASET_H_
#define _KITTI_DATASET_H_

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <tuple>
#include "./../ara_slam.h"
#include "./../calibration/calibrator.h"
#include "dataset.h"

namespace ara_slam{ namespace data {
    /**
     * 
     * */
    class KittiDataset: public Dataset<std::tuple<cv::Mat,cv::Mat>> {
    private:
        std::vector<std::tuple<std::string,std::string>> stereo_images;
        std::string bucket;
        std::string data_dir;

    public:       
        KittiDataset(std::string data_dir, std::string bucket);
        calibration::stereo_t calibration_params();
        int length();
        /**
         * TODO: Implement a function that reads the trajectory file and plots it in an image
         * 
         * */
        cv::Mat trajectory();
        
        /**
         * Returns a tuple of cv::Mat, as the dataset should not be changed, therefore 
         * an actual object is returned
         * */
        std::tuple<cv::Mat,cv::Mat> operator[](int);
    };
}}

#endif