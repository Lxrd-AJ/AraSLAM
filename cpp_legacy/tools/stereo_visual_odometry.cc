#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/sfm/projection.hpp>
#include <vector>
#include <boost/range/iterator_range.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include "ara_slam.h"
#include "calibration/calibrator.h"
#include "depth/depth_map.h"
#include "features/features.h"
#include "localisation/localisation.h"
#include "viz/viz.h"

namespace fs = boost::filesystem;
namespace as = ara_slam;
namespace calib = ara_slam::calibration;
namespace depth = ara_slam::depth;
namespace feats = ara_slam::features;
namespace viz = ara_slam::viz;
namespace local = ara_slam::localisation;

//TODO: Make sure this is done using a pipeline component based approach like ampnet/OpenCV graph API
/**
 * TODO: ELP Camera Calibration
 * See also https://albertarmea.com/post/opencv-stereo-camera/
 * 1. Use full size image (only resize and change intrinsic params after)
 * 2. Use about 40 calibration patterns
 * */

/**
 * 
 * Lazy loading. Return a list of the strings only
 * */
std::vector<std::tuple<std::string,std::string>> load_kitti_stereo(std::string ref){
    const auto iterator = fs::directory_iterator(ref + "/image_0");
    std::vector<std::tuple<std::string,std::string>> stereo_images;
    std::vector<std::string> names;
    for(const auto &entry: boost::make_iterator_range(iterator,{})){
        auto filename = entry.path().filename().string();
        names.push_back( filename );
        std::cout <<  "-> Kitti image " << filename << std::endl;
    }
    std::sort(names.begin(), names.end());
    for(const auto &name: names){
        auto stereo_pair = std::make_tuple( 
            ref + "/image_0/" + name, 
            ref + "/image_1/" + name
        );
        stereo_images.push_back( stereo_pair );
    }
    return stereo_images;
}

calib::stereo_t load_kitti_calibration(std::string ref){ 
    //P0 = left_grayscale, P1 = right_grayscale_camera, P2 = left color cam, P3 = right color cam   
    std::string filename = ref + "/calib.txt";
    std::ifstream calib_file(filename);
    std::vector<std::string> lines;    
    std::vector<cv::Mat> pmtxs;
    std::string line;
    std::cout << "\nReading calibration file : " << filename << std::endl;
    while( std::getline(calib_file,line) ){ lines.push_back(line); }
    for(const auto x: lines){         
        std::vector<std::string> str_mat;
        std::vector<double> proj_mat;
        boost::split(str_mat,x,[](char c){ return c == ':'; });        
        boost::split(str_mat,str_mat[1],[](char c){ return c == ' '; }); 
        str_mat.erase(str_mat.begin());        
        std::transform(str_mat.begin(), str_mat.end(), std::back_inserter(proj_mat), [](auto s) -> double {
            return std::stod(s);
        });
        cv::Mat mtx = cv::Mat(proj_mat,true).reshape(1,3);
        pmtxs.push_back(mtx);
    }
    //populate the disparity to depth    
    cv::Mat Q = cv::Mat::zeros(4,4,CV_64F);
    cv::Mat K,R,t;
    cv::sfm::KRtFromProjection(pmtxs[0], K,R,t ); //uses RQ decomposition
    K *= 0.5; //because the images are resized by half
    Q.at<double>(0,0) = 1.0; Q.at<double>(0,3) = -K.at<double>(0,2);
    Q.at<double>(1,1) = 1.0; Q.at<double>(1,3) = -K.at<double>(1,2);
    Q.at<double>(2,3) = K.at<double>(0,0);
    Q.at<double>(3,2) = -1.0/0.54; //baseline is 54cm/0.54m taken from http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf    
    calib::stereo_t kitti_stereo = {
        .left_projection = pmtxs[0], 
        .right_projection = pmtxs[1],
        .disparity_depth = Q
    };
    kitti_stereo.left.intrinsic.cam_matrix = K;
    kitti_stereo.left.extrinsic.rotation_matrix = R;
    kitti_stereo.left.extrinsic.translation_vec = t;
    return kitti_stereo;
}

std::tuple<cv::Mat,cv::Mat> read_stereo_pair(std::tuple<std::string,std::string> item){
    cv::Mat left = cv::imread(std::get<0>(item));
    cv::Mat right = cv::imread(std::get<1>(item));
    cv::resize(left, left, cv::Size(), 0.50, 0.50);
    cv::resize(right, right, cv::Size(), 0.50, 0.50);
    return std::make_tuple(left, right);
}


int main(int argc, char *argv[]){
    auto bucket = "./data/data_odometry_gray/dataset/sequences/00";
    auto dataset = load_kitti_stereo(bucket);
    calib::stereo_t kitti_stereo = load_kitti_calibration(bucket); 
    
    std::cout << "Total frames read " << dataset.size() << std::endl;

    cv::Mat keypoint_tracks;
    std::tuple<cv::Mat,cv::Mat> previous_item;

    const std::string _KEYPOINT_POINTCLOUD_ = "KeyPoint Point Cloud";
    const std::string _IMAGE_POINTCLOUD_ = "Global Image Point Cloud";
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Image PCL viewer"));
    pcl::visualization::PCLVisualizer::Ptr kp_viewer (new pcl::visualization::PCLVisualizer ("Keypoint PCL viewer"));
    pcl::PointCloud<pcl::PointXYZ>::Ptr kp_pcl (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr img_pcl (new pcl::PointCloud<pcl::PointXYZ>);
    viewer->addPointCloud<pcl::PointXYZ> (img_pcl, _IMAGE_POINTCLOUD_);
    kp_viewer->addPointCloud<pcl::PointXYZ> (kp_pcl, _KEYPOINT_POINTCLOUD_);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _IMAGE_POINTCLOUD_);
    viewer->addCoordinateSystem(0.5);
    kp_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _KEYPOINT_POINTCLOUD_);
    kp_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1,1,0, _KEYPOINT_POINTCLOUD_);
    kp_viewer->addCoordinateSystem (1.0); 

    std::vector<std::tuple<cv::Mat,cv::Mat>> odometry; //tuple of rotation and translation
    cv::Mat acc_R = cv::Mat::eye(3,3,CV_64F); //accumulator for rotation
    cv::Mat acc_t = cv::Mat::zeros(3,1,CV_64F); //translation
    cv::Mat trajectory = cv::Mat::zeros(500,500, CV_8UC3); //visualisation    
    int counter = -1;
    for(const auto &item: dataset){
        ++counter;
        if(counter == 0){
            previous_item = read_stereo_pair(item);
            keypoint_tracks = cv::Mat::zeros(std::get<0>(previous_item).size(),CV_8UC3);
            continue;
        }
        
        cv::Mat cur_left_img, cur_right_img, old_left_img, old_right_img;
        std::tie(cur_left_img, cur_right_img) = read_stereo_pair(item);
        std::tie(old_left_img, old_right_img) = previous_item;

        // cv::imshow("Stereo",as::make_anaglyph(std::get<0>(*previous_item),std::get<1>(*previous_item)));
        
        //Show previous, current and depth image
        //data type of depth_map is CV_16UC1 but converted to CV_8UC1
        cv::Mat depth_map = depth::compute_depth(cur_left_img, cur_right_img);            
        cv::imshow("Left Feed", cur_left_img);
        // depth_map.convertTo(depth_map, CV_8U, (1/256));
        // cv::Mat viz_depth_map;
        // cv::ximgproc::getDisparityVis(depth_map, viz_depth_map, 3.5);
        cv::imshow("Depth Map", depth_map);        
        
        //extract keypoints from the previous and current image
        auto fcl_kps = feats::extract_features(cur_left_img);
        // auto fcr_kps = std::get<0>(feats::extract_features(cur_right_img, as::feature_extractor::fast));
        auto fol_kps = feats::extract_features(old_left_img);//as::feature_extractor::sift

        //Show the keypoints for viz purposes
        cv::Mat img_fcl_kps = cur_left_img.clone();
        cv::drawKeypoints(cur_left_img, std::get<0>(fcl_kps), img_fcl_kps, cv::Scalar(100,50,200));
        cv::imshow("Left keypoints", img_fcl_kps);

        //track the keypoints across the prev image and the current image using only the left image        
        std::vector<cv::DMatch> matches = feats::compare_features(std::get<1>(fol_kps),std::get<1>(fcl_kps));
        //show the matches
        cv::Mat viz_matches;
        cv::drawMatches(old_left_img, std::get<0>(fol_kps), cur_left_img, std::get<0>(fcl_kps), matches, viz_matches);
        cv::imshow("Prev frame - current frame matches", viz_matches);
        
        std::vector<cv::KeyPoint> current_kp; //filtered current frame keypoints
        std::vector<cv::KeyPoint> previous_kp; //filtered previous frame keypoints
        std::for_each(matches.begin(), matches.end(), [&current_kp, &previous_kp, fcl_kps, fol_kps](cv::DMatch match){
            current_kp.push_back( std::get<0>(fcl_kps)[match.trainIdx] );
            previous_kp.push_back( std::get<0>(fol_kps)[match.queryIdx]);
        });
        //show the filtered current frame features
        cv::Mat viz_fil_kp = cur_left_img.clone();
        cv::drawKeypoints(cur_left_img, current_kp, viz_fil_kp, cv::Scalar(100,50,200));
        cv::imshow("Filtered Keypoints", viz_fil_kp);

        //draw the current image as a point cloud
        viewer->removePointCloud(_IMAGE_POINTCLOUD_);
        cv::Mat _3dimage;//CV_32FC3
        cv::reprojectImageTo3D(depth_map, _3dimage, kitti_stereo.disparity_depth,true);
        std::vector<cv::Vec3f> viz_pcl(_3dimage.begin<cv::Vec3f>(), _3dimage.end<cv::Vec3f>());
        img_pcl = viz::make_pointcloud(viz_pcl);
        viewer->addPointCloud<pcl::PointXYZ> (img_pcl, _IMAGE_POINTCLOUD_);        

        //Get the 3D points of the features and show them
        // kp_viewer->removePointCloud(_KEYPOINT_POINTCLOUD_);
        std::vector<cv::Point3f> point_cl = depth::world_xyz_from_keypoints(current_kp,_3dimage);
        std::vector<cv::Point3f> prev_point_cl = depth::world_xyz_from_keypoints(previous_kp,_3dimage);         
        std::vector<cv::Point3f> local_point_cl;
        std::transform(point_cl.begin(), point_cl.end(), std::back_inserter(local_point_cl), 
            [acc_R, acc_t](cv::Point3f p) -> cv::Point3f {
                cv::Mat x = cv::Mat(p); //3x1
                x.convertTo(x,CV_64F);                
                cv::Mat y = (acc_R * x) + acc_t;
                return cv::Point3f(y.at<float>(0,0),y.at<float>(1,0),y.at<float>(2,0));
        });
        // std::cout << local_point_cl << std::endl;
        kp_pcl = viz::make_pointcloud(local_point_cl);
        kp_viewer->updatePointCloud(kp_pcl, _KEYPOINT_POINTCLOUD_);
        
        
        //Detect the inliers between the previous and current point clouds
        // std::vector<cv::Point3f> inliers = local::max_clique(prev_point_cl, point_cl);
        std::vector<int> clique = local::max_clique(prev_point_cl, point_cl);
        std::cout << "Found " << clique.size() << " inliers" << std::endl;
        if( clique.size() < 4 ){ continue; } //the optimisation algorithm requires at least 4 points
        //TODO: Investigate if the order of the clique matches up with the order of the keypoints
        std::vector<cv::Point3f> world_points;
        std::vector<cv::Point2f> image_points;
        std::for_each(clique.begin(), clique.end(), [point_cl, current_kp, &world_points, &image_points](int idx) -> void {
            world_points.push_back(point_cl[idx]);
            image_points.push_back(current_kp[idx].pt);
        });
        //Alternate world and image points generation
        // std::vector<cv::Point3f> world_points;
        // std::vector<cv::Point2f> image_points;
        // for(size_t idx=0; idx < current_kp.size(); idx++){
        //     world_points.push_back( point_cl[idx] );
        //     image_points.push_back(current_kp[idx].pt);
        // }
        
        cv::Mat K = kitti_stereo.left.intrinsic.cam_matrix;
        cv::Mat dist = kitti_stereo.left.intrinsic.distortion;
        cv::Mat R, t;
        bool result = cv::solvePnP(world_points, image_points, K, dist, R, t);
        if( !result ){
            std::cout << "***Failed to detect camera pose" << std::endl;
            continue;
        }
        cv::Rodrigues(R,R);
        kitti_stereo.left.extrinsic.rotation_matrix = R;
        kitti_stereo.left.extrinsic.translation_vec = t;
        odometry.push_back( std::make_tuple(R,t) );
        if((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))){
            //if dominant forward motion
            acc_t = acc_t + (acc_R * t);
            acc_R = R * acc_R;
        }
        std::cout << "[Accumulative coords] x = " << acc_t.at<double>(0,0) 
                  << " y = " << acc_t.at<double>(1,0) 
                  << " z = " << acc_t.at<double>(2,0) << std::endl;


        //As the camera is moving, compute the current R and t and use that to obtain the 2D image points        
        // visualise the inliers
        std::vector<cv::Point2i> inliers = depth::world_xyz_to_image(world_points, kitti_stereo);
        cv::Mat viz_inliers = cur_left_img.clone();
        for(const cv::Point2i p: inliers){                        
            cv::circle(viz_inliers, cv::Point(p.x,p.y), 2, cv::Scalar(50,200,100));
        }
        cv::imshow("Inliers", viz_inliers);


        //Plot the trajectory 2D
        int coord_x = (int)acc_t.at<double>(0,0) + 250;
        int coord_z = (int)acc_t.at<double>(2,0) + 250;
        cv::circle(trajectory, cv::Point(coord_x,coord_z), 2, cv::Scalar(10,10,255));
        cv::imshow("Trajectory", trajectory);



        viewer->spinOnce(10);
        kp_viewer->spinOnce(10);
        cv::waitKey(100);
        previous_item = std::make_tuple(cur_left_img, cur_right_img);
    }
    return 0;
}
