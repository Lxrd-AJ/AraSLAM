#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>
// #include <opencv2/sfm/projection.hpp>
#include <tuple>
#include <chrono>
#include "ara_slam.h"
#include "visual_odometry.h"
#include "data/kitti_dataset.h"
#include "data/video_dataset.h"
#include "calibration/calibrator.h"
#include "features/features.h"
#include "localisation/localisation.h"
#include "viz/viz.h"

namespace fs = boost::filesystem;
namespace as = ara_slam;
namespace calib = ara_slam::calibration;
// namespace depth = ara_slam::depth;
namespace feats = ara_slam::features;
namespace viz = ara_slam::viz;
namespace local = ara_slam::localisation;
namespace data = ara_slam::data;

int main(int argc, char *argv[]){
    auto bucket = "02"; //./data/data_odometry_gray/dataset/sequences/
    auto dir = "./data/data_odometry_gray/dataset";
    data::KittiDataset dataloader = data::KittiDataset(dir,bucket);

    calib::stereo_t kitti_stereo = dataloader.calibration_params();
    const cv::Mat K = kitti_stereo.left.intrinsic.cam_matrix;
    
    pcl::visualization::PCLVisualizer::Ptr viewer = viz::init_pclviewer("VO");
    as::VisualOdometry viz_odometry = as::VisualOdometry(K);

    for(size_t idx = 0; idx < dataloader.length(); ++idx){
        std::tuple<cv::Mat,cv::Mat> item = dataloader[idx];
        cv::Mat frame = std::get<0>(item); //this is the left frame
        
        cv::imshow("Current frame ", frame);

        if( viz_odometry.state() == as::odometry_state::initialising ){
            bool status = viz_odometry.initialise(frame);
        }

        viewer->spinOnce(10);
        cv::waitKey(1);
    }

	return 0;

}