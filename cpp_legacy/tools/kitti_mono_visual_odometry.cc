#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>
// #include <opencv2/sfm/projection.hpp>
#include <tuple>
#include <chrono>
#include "data/kitti_dataset.h"
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
    // auto bucket = "./data/data_odometry_gray/dataset/sequences/04";
    data::KittiDataset dataloader = data::KittiDataset("./data/data_odometry_gray/dataset","00");
    calib::stereo_t kitti_stereo = dataloader.calibration_params();
    const cv::Mat K = kitti_stereo.left.intrinsic.cam_matrix;

    std::cout << "Processing " << dataloader.length() << " frames" << std::endl;

    cv::Mat previous_frame;
    
    std::vector<cv::Point2f> prev_features;
    std::vector<cv::Mat> acc_tracked_fts;
    std::vector<cv::Mat> acc_proj_mtx;
    std::vector<std::tuple<cv::Mat,cv::Mat>> odometry; //tuple of rotation and translation
    
    cv::Mat acc_R = cv::Mat::eye(3,3,CV_64F); //accumulator for rotation
    cv::Mat acc_t = cv::Mat::zeros(3,1,CV_64F); //translation
    cv::Mat trajectory = cv::Mat::zeros(1000, 1000, CV_8UC3);
    // cv::Mat trajectory = dataloader.trajectory();
    
    const std::string _IMAGE_POINTCLOUD_ = "Global Image Point Cloud";
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Image PCL viewer"));
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
    viewer->addPointCloud<pcl::PointXYZ> (point_cloud, _IMAGE_POINTCLOUD_);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _IMAGE_POINTCLOUD_);
    viewer->addCoordinateSystem(0.5);

    const int MIN_NUM_FEAT = 500;

    for(size_t idx = 0; idx < dataloader.length(); ++idx){
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::tuple<cv::Mat,cv::Mat> item = dataloader[idx];
        cv::Mat frame = std::get<0>(item); //this is the left frame
        cv::imshow("Current frame", frame);

        if( previous_frame.empty() ){
            previous_frame = frame;
            std::vector<cv::KeyPoint> previous_kps = std::get<0>(feats::extract_features(previous_frame, as::feature_extractor::fast));
            cv::KeyPoint::convert(previous_kps,prev_features);
            continue;
        }
                
        //track the keypoints
        auto filtered_kps = local::track_keypoints(previous_frame, prev_features, frame);
        std::vector<cv::Point2f> filtered_prev_features = std::get<0>(filtered_kps);
        std::vector<cv::Point2f> tracked_features = std::get<1>(filtered_kps);
        //visualise the keypoints
        std::vector<cv::KeyPoint> current_kps;
        cv::KeyPoint::convert(tracked_features, current_kps);
        cv::Mat img_keypoints = frame.clone();
        cv::drawKeypoints(frame, current_kps, img_keypoints, cv::Scalar(80,20,210));
        cv::imshow("Tracked keypoints", img_keypoints);
        
        //Recover the pose from the points
        auto pose = local::recover_pose(tracked_features, filtered_prev_features, K);
        cv::Mat R = std::get<0>(pose);
        cv::Mat t = std::get<1>(pose);            
        odometry.push_back( std::make_tuple(R,t) );

        if((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))){
            //if dominant forward motion
            acc_t = acc_t + (acc_R * t);
            acc_R = R * acc_R;
        }        
        int coord_x = (int)acc_t.at<double>(0,0) + 500;
        int coord_z = (int)acc_t.at<double>(2,0) + 300;
        cv::circle(trajectory, cv::Point(coord_x,coord_z), 2, cv::Scalar(10,10,255));
        cv::imshow("Trajectory", trajectory);

        cv::Mat P;
        cv::sfm::projectionFromKRt(K,acc_R,acc_t,P);                        
        
        previous_frame = frame.clone();
        prev_features = tracked_features;

        //use the fast algorithm to detect features in the current frame until they drop below a threshold
        if( prev_features.size() < MIN_NUM_FEAT ){            
            std::cout << "Tracked points fallen below threshold (" << prev_features.size() << ")" << std::endl;
            std::vector<cv::KeyPoint> previous_kps = std::get<0>(feats::extract_features(previous_frame, as::feature_extractor::fast));
            cv::KeyPoint::convert(previous_kps,prev_features);
            std::cout << "Triggering a re-detection" << std::endl;
        }
        
        viewer->spinOnce(10);
        cv::waitKey(10);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Iteration " << idx << " completed in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" <<  std::endl;
    }    

    return 0;

}