#ifndef _VIDEO_DATASET_H_
#define _VIDEO_DATASET_H_

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
    class VideoDataset: public Dataset<cv::Mat> {
    private:        
        std::string filename;
		float fps;
		int frame_count;
		cv::VideoCapture capture;

    public:       
        VideoDataset(std::string ref);        
		~VideoDataset();
        int length();
		float duration();
		cv::Mat stream();
                
        cv::Mat operator[](int);
    };
}}

#endif