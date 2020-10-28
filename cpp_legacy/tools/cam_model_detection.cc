#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <string>
#include "ara_slam.h"
#include "calibration/calibrator.h"
#include "features/features.h"
// #include <yaml-cpp/yaml.h>

namespace fs = boost::filesystem;
namespace as = ara_slam;
namespace calib = ara_slam::calibration;
namespace desc = ara_slam::features;

int main(int argc, char *argv[]){
    const cv::Size board_size(8,6);
    const float square_size = 15.0f;
    std::vector<cv::Mat> images;
    cv::Mat axis = cv::Mat::zeros(3,3,CV_64F);
    const std::string data_path = "./data/monocular";
    const auto iterator = fs::directory_iterator(data_path);
    
    // for(const auto &entry: boost::make_iterator_range(iterator,{})){
    //     std::cout << "Reading image " << entry << std::endl;
    //     images.push_back( cv::imread(entry.path().string()) );
    // }

    // calib::camera_matrix_t camera_matrix = calib::calibrate_camera(images, board_size, square_size);
    // std::cout << "Finished calibrating the camera" << std::endl;


    // //Read from the mesh directory
    // const std::string mesh_dir = "./data/frag_mesh";
    // const auto mesh_iter = fs::directory_iterator(mesh_dir);
    // images.clear(); //empty the list as we're re-using it
    // std::vector<desc::features_t> features;
    // for(const auto &entry: boost::make_iterator_range(mesh_iter,{})){
    //     std::cout << "Reading mesh " << entry << std::endl;
    //     cv::Mat image = cv::imread(entry.path().string());
    //     desc::features_t res = desc::extract_features(image);
    //     images.push_back( image );
    //     features.push_back( res );
    // }   




    //Read from the camera
    cv::VideoCapture stereo_capture(1);

    while( !stereo_capture.isOpened() ){
        std::cout << "Waiting for stereo camera" << std::endl;
    }

    // stereo_capture.set(CV_CAP_PROP_FRAME_WIDTH, 1440); //1280
    // stereo_capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720); //720

    int counter = 0;
    do{
        cv::Mat cur_frame;
        stereo_capture >> cur_frame;

        //Resize and change the intrinsic parameters of the camera
        cv::resize(cur_frame, cur_frame, cv::Size(), 0.50, 0.50);
        // camera_matrix.intrinsic.cam_matrix *= 0.5;

        cv::Mat left_frame = cur_frame(cv::Rect(0, 0, cur_frame.cols/2, cur_frame.rows));

        if( cur_frame.empty() ){
            std::cout << "Empty frame read .." << std::endl;
            continue;
        }
        if( (char)cv::waitKey(25) == 27 )
            break;
        if ((char)cv::waitKey(20) == 's'){
            std::cout << "Saving image " << counter << std::endl;
            cv::imwrite("./data/stereo/" + std::to_string(counter) + ".jpg", cur_frame);
        }
        if(counter % 100 == 0){
            std::cout << "Saving image " << counter << std::endl;
            cv::imwrite("./data/stereo/" + std::to_string(counter) + ".jpg", cur_frame);
        }
        counter++;

        cv::imshow("Live Feed", cur_frame);
        std::cout << "Size = " << cur_frame.size() << std::endl;
        // std::cout << "Intrinsic params = " << camera_matrix.intrinsic.cam_matrix << std::endl;
 
        //for each image in our list, compare for matches against the current live image
        // for(const auto &image: images){
        //     int idx = &image - &images[0];
        //     desc::features_t cur_feats = features[idx];
        //     desc::features_t hot_feats = desc::extract_features(left_frame);
        //     auto cur_key_pts = std::get<0>(cur_feats);
        //     auto hot_key_pts = std::get<0>(hot_feats);
        //     auto matches = desc::compare_features(std::get<1>(cur_feats), std::get<1>(hot_feats));
        //     cv::Mat match_view;
        //     cv::drawMatches(image, cur_key_pts, left_frame, hot_key_pts, matches, match_view);
        //     cv::imshow("View " + std::to_string(idx), match_view);
        // }

    }while( stereo_capture.isOpened() );

    stereo_capture.release();
    cv::destroyAllWindows();

    return 0;
}