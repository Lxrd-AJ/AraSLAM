#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <string>
#include "ara_slam.h"
#include "calibration/calibrator.h"
// #include <yaml-cpp/yaml.h>

namespace fs = boost::filesystem;
namespace as = ara_slam;
namespace calib = ara_slam::calibration;
using namespace std::placeholders;

int main(int argc, char *argv[]){
    bool STEREO = true;
    const cv::Size board_size(8,6);
    const float square_size = 15.0f;
    
    
    if( STEREO ){ //Perform stereo calibration        
        const std::string data_path = "./data/stereo";
        const auto iterator = fs::directory_iterator(data_path);
        std::cout << "Stereo calibration " << std::endl;
        std::vector<std::tuple<cv::Mat,cv::Mat>> stereo_images;
        for(const auto &entry: boost::make_iterator_range(iterator,{})){
            cv::Mat image = cv::imread(entry.path().string());            
            std::cout << "Reading image " << entry << " with size = " << image.size() << std::endl;
            cv::Mat left_image = image(cv::Rect(0, 0, image.cols/2, image.rows));
            cv::Mat right_image = image(cv::Rect(image.cols/2, 0, image.cols/2, image.rows));            
            stereo_images.push_back( std::make_tuple(left_image, right_image));
        }

        calib::stereo_t stereo_data = calib::calibrate_stereo_camera(stereo_images, board_size, square_size);

        std::cout << "Left Intrinsic matrix " << stereo_data.left.intrinsic.cam_matrix << std::endl;
        std::cout << "Right Intrinsic matrix " << stereo_data.right.intrinsic.cam_matrix << std::endl;
        std::cout << "Left distortion " << stereo_data.left.intrinsic.distortion << std::endl;
        std::cout << "Right distortion " << stereo_data.right.intrinsic.distortion << std::endl;

        int counter = 0;
        for(const auto &stereo_pair: stereo_images){
            cv::Mat l_undistort = calib::reproject_image( stereo_data.left.intrinsic, std::get<0>(stereo_pair) );
            cv::Mat r_undistort = calib::reproject_image( stereo_data.right.intrinsic, std::get<1>(stereo_pair) );
            auto rect_stereo_pair = calib::stereo_rectify( stereo_pair, stereo_data );            
            cv::Mat comb_undistort_stereo_img = as::hconcat( l_undistort, r_undistort );
            cv::Mat comb_stereo_image = as::hconcat( std::get<0>(stereo_pair), std::get<1>(stereo_pair));
            cv::Mat comb_rect_stereo = as::hconcat( std::get<0>(rect_stereo_pair), std::get<1>(rect_stereo_pair) );

            cv::imshow( "Original stereo pair", comb_stereo_image );
            cv::imshow("Undistorted stereo", comb_undistort_stereo_img);
            cv::imshow("Rectified stereo", comb_rect_stereo);            

            std::string st_image_path = "./.tmp/stereo_image_" + std::to_string(counter) + ".jpg";
            std::string st_und_path = "./.tmp/stereo_undistorted_" + std::to_string(counter) + ".jpg";
            std::string st_rect_path = "./.tmp/stereo_rectified_" + std::to_string(counter) + ".jpg";
            cv::imwrite(st_image_path, comb_stereo_image);
            cv::imwrite(st_und_path, comb_undistort_stereo_img);
            cv::imwrite(st_rect_path, comb_rect_stereo);

            cv::waitKey(2000);
            ++counter;
        }

    }else{        
        std::vector<cv::Mat> images;
        const std::string data_path = "./data/monocular";
        const auto iterator = fs::directory_iterator(data_path);
        
        for(const auto &entry: boost::make_iterator_range(iterator,{})){
            std::cout << "Reading image " << entry << std::endl;
            images.push_back( cv::imread(entry.path().string()) ); //, cv::IMREAD_GRAYSCALE
        }

        calib::camera_matrix_t camera_matrix = calib::calibrate_camera(images, board_size, square_size);
        std::cout << "Intrinsic matrix \n" << camera_matrix.intrinsic.cam_matrix << std::endl;
        std::cout << "\nDistortion Coefficients \n" << camera_matrix.intrinsic.distortion << std::endl;

        double error = calib::calculate_projection_error(camera_matrix.intrinsic);

        std::vector<cv::Mat> reproj_images(images.size());
        auto func = std::bind(calib::reproject_image, camera_matrix.intrinsic, _1);
        std::transform(images.begin(), images.end(), reproj_images.begin(), func);

        int counter = 0;
        std::vector<float> mat_holder;
        // YAML::Emitter yaml_out;
        // yaml_out << "camera poses";
        
        for(const auto &image: images){
            cv::Mat combined_image;
            cv::Mat reprojected_image = calib::reproject_image(camera_matrix.intrinsic, image);        
            std::string calib_image_path = "./.tmp/calibrated_image_" + std::to_string(counter) + ".jpg";
            std::string calib_image_joined_path = "./.tmp/calibrated_image_joined_" + std::to_string(counter) + ".jpg";        
            cv::hconcat(image, reprojected_image, combined_image);
            cv::imwrite(calib_image_path, reprojected_image);
            cv::imwrite(calib_image_joined_path, combined_image);
            cv::imshow("Distorted (L) and Undistorted (R) image", combined_image);
            cv::waitKey(100);
            calib::extrinsic_t pose = calib::get_camera_chessboard_pose(board_size, square_size, camera_matrix.intrinsic, image);
            // std::cout << "Rotation:\n" << pose.rotation_matrix << std::endl;
            // std::cout << "Translation\n" << pose.translation_vec << std::endl;

            // yaml_out << YAML::BeginMap;
            // yaml_out << YAML::Key << "img_name";
            // yaml_out << YAML::Value << calib_image_path;
            // yaml_out << YAML::Key << "rotation";
            
            pose.rotation_matrix.col(0).copyTo(mat_holder);
            // yaml_out << YAML::Value << YAML::BeginSeq << mat_holder << YAML::EndSeq;
            // yaml_out << YAML::Key << "translation";
            pose.translation_vec.col(0).copyTo(mat_holder);
            // yaml_out << YAML::Value << YAML::BeginSeq << mat_holder << YAML::EndSeq;
            // yaml_out << YAML::EndMap;

            counter++;
        }
        

        
        // std::cout << "Here's the output YAML:\n" << yaml_out.c_str();
        // std::ofstream fout(".tmp/file.yaml");
        // fout << yaml_out.c_str();
    }

    return 0;
}