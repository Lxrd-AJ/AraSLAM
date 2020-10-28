#include <opencv2/opencv.hpp>
#include <opencv2/sfm/projection.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <tuple>
#include <vector>
#include <fstream>
#include <limits>
#include "./../ara_slam.h"
#include "dataset.h"
#include "kitti_dataset.h"


using namespace boost::filesystem;

namespace ara_slam{ namespace data {    

    int KittiDataset::length(){
        return stereo_images.size();
    }

    std::tuple<cv::Mat,cv::Mat> KittiDataset::operator[](int index){
        if(index > stereo_images.size()){
            std::cout << "Index out of bounds" << std::endl;
            exit(0);
        }
        auto item = stereo_images[index];
        cv::Mat left = cv::imread(std::get<0>(item));
        cv::Mat right = cv::imread(std::get<1>(item));
        // cv::resize(left, left, cv::Size(), 0.50, 0.50);
        // cv::resize(right, right, cv::Size(), 0.50, 0.50);
        return std::make_tuple(left, right);
    }
    
    KittiDataset::KittiDataset(std::string data_dir, std::string bucket)
        : bucket(bucket), data_dir(data_dir){        
        std::string ref = data_dir + "/sequences/" + bucket;
        const auto iterator = directory_iterator(ref + "/image_0");        
        std::vector<std::string> names;
        //use the image_0 folder to get all the filenames
        for(const auto &entry: boost::make_iterator_range(iterator,{})){
            auto filename = entry.path().filename().string();
            names.push_back( filename );            
        }
        std::sort(names.begin(), names.end());
        for(const auto &name: names){
            auto stereo_pair = std::make_tuple( 
                ref + "/image_0/" + name, 
                ref + "/image_1/" + name
            );
            stereo_images.push_back( stereo_pair );
        }
    }

    /**
     * TODO: Fix the bug in the trajectory function 
     * */
    cv::Mat KittiDataset::trajectory(){
        std::string file = data_dir + "/poses/" + bucket + ".txt";
        std::ifstream poses(file);
        std::vector<std::string> lines; 
        std::string line;        
        while( std::getline(poses,line) ){ lines.push_back(line); }
        cv::Mat acc_R = cv::Mat::eye(3,3,CV_64F);
        cv::Mat acc_t = cv::Mat::zeros(3,1,CV_64F);
        int min_x, max_x, min_y, max_y;
        cv::Mat trajectory = cv::Mat::zeros(50, 50, CV_8UC3);
        
        for(const auto x: lines){
            std::vector<std::string> str_pose;
            boost::split(str_pose,x,[](char c){ return c == ' '; });
            std::vector<double> rotation;
            std::vector<double> translation;
            std::transform(str_pose.begin(), str_pose.end() - 3, std::back_inserter(rotation), [](auto s) -> double {
                return std::stod(s);
            });
            std::transform(str_pose.end()-3, str_pose.end(), std::back_inserter(translation), [](auto s) -> double {
                return std::stod(s);
            });
            cv::Mat R = cv::Mat(3,3,CV_64F,rotation.data());
            cv::Mat t = cv::Mat(3,1,CV_64F,translation.data());

            // if((t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))){
            // }
            acc_t = acc_t + (acc_R * t);
            acc_R = R * acc_R;
            
            // std::cout << acc_R << std::endl;
            // std::cout << acc_t << std::endl;
            int coord_x = (int)acc_t.at<double>(0,0);
            int coord_y = (int)acc_t.at<double>(2,0); //the z-coord 

            if(coord_y == std::numeric_limits<int>::min() || coord_y == std::numeric_limits<int>::max()){
                continue;
            }
            if(coord_x == std::numeric_limits<int>::min() || coord_x == std::numeric_limits<int>::max()){
                continue;
            }
            
            cv::Mat pad = cv::Mat::zeros(trajectory.size(),CV_8UC3);
            if( coord_x > trajectory.cols ){                                            
                trajectory = ara_slam::hconcat(trajectory,pad);                            
            }else if( coord_x < 0 ){
                trajectory = ara_slam::hconcat(pad,trajectory);                
            }
            // std::cout << coord_x << std::endl;
            // std::cout << trajectory.cols << std::endl;
            // std::cout << coord_y << std::endl;
            // std::cout << trajectory.rows << std::endl;
            if( coord_y > trajectory.rows ){                                            
                trajectory = ara_slam::vconcat(trajectory,pad);
            }else if( coord_y < 0 ){
                trajectory = ara_slam::vconcat(pad,trajectory);
            }
            
            cv::circle(trajectory, cv::Point(coord_x,coord_y), 2, cv::Scalar(10,255,10));
        }
        
        return trajectory;
    }

    calibration::stereo_t KittiDataset::calibration_params(){
        //P0 = left_grayscale, P1 = right_grayscale_camera, P2 = left color cam, P3 = right color cam   
        std::string filename = data_dir + "/sequences/" + bucket + "/calib.txt";
        std::ifstream calib_file(filename);
        std::vector<std::string> lines;    
        std::vector<cv::Mat> pmtxs;
        std::string line;        
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
        Q.at<double>(0,0) = 1.0; Q.at<double>(0,3) = -K.at<double>(0,2);
        Q.at<double>(1,1) = 1.0; Q.at<double>(1,3) = -K.at<double>(1,2);
        Q.at<double>(2,3) = K.at<double>(0,0);
        Q.at<double>(3,2) = -1.0/0.54; //baseline is 54cm/0.54m taken from http://ww.cvlibs.net/publications/Geiger2013IJRR.pdf    
        calibration::stereo_t kitti_stereo = {
            .left_projection = pmtxs[0], 
            .right_projection = pmtxs[1],
            .disparity_depth = Q
        };
        kitti_stereo.left.intrinsic.cam_matrix = K;
        kitti_stereo.left.extrinsic.rotation_matrix = R;
        kitti_stereo.left.extrinsic.translation_vec = t;
        return kitti_stereo;
    }
}}