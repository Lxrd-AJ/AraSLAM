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
    const cv::Size board_size(8,6);
    const float square_size = 15.0f;
    std::vector<cv::Mat> images;
    cv::Mat axis = cv::Mat::zeros(3,3,CV_64F);
    const std::string data_path = "./data/monocular";
    const auto iterator = fs::directory_iterator(data_path);
    
    for(const auto &entry: boost::make_iterator_range(iterator,{})){
        std::cout << "Reading image " << entry << std::endl;
        images.push_back( cv::imread(entry.path().string()) ); //, cv::IMREAD_GRAYSCALE
    }

    calib::detected_corners_t det_corners = calib::detect_corners(images, board_size, square_size);
    std::vector<std::vector<cv::Point3f>> world_corners = std::get<0>(det_corners);
    std::vector<std::vector<cv::Point2f>> chess_corners = std::get<1>(det_corners);
    calib::camera_matrix_t camera_matrix = calib::calibrate_camera(images, board_size, square_size);

    axis.at<double>(0,0) = 3 * square_size;
    axis.at<double>(1,1) = 3 * square_size;
    axis.at<double>(2,2) = -3 * square_size;

    
    for(const auto &image: images){
        int index = &image - &images[0];
        std::vector<cv::Point2f> corners = chess_corners[index];
        cv::Mat _image = calib::reproject_image(camera_matrix.intrinsic, image);
        calib::extrinsic_t pose = calib::get_camera_chessboard_pose(board_size, square_size, camera_matrix.intrinsic, image, calib::rotation_format::rotation_matrix);
        cv::Mat image_points;
        cv::projectPoints(axis, pose.rotation_matrix, pose.translation_vec, 
                    camera_matrix.intrinsic.cam_matrix, camera_matrix.intrinsic.distortion,
                    image_points 
        );

        // std::cout << "Index " << index << std::endl;
        // std::cout << "Image points " << image_points << std::endl;

        cv::Point x_line = cv::Point(image_points.at<double>(0,0), image_points.at<double>(0,1));
        cv::Point y_line = cv::Point(image_points.at<double>(1,0), image_points.at<double>(1,1));
        cv::Point z_line = cv::Point(image_points.at<double>(2,0), image_points.at<double>(2,1));
        cv::line(image, corners[0], x_line, cv::Scalar(255,0,0), 2.0f);
        cv::line(image, corners[0], y_line, cv::Scalar(0,255,0), 2.0f);
        cv::line(image, corners[0], z_line, cv::Scalar(0,0,255), 2.0f);

        cv::imshow("image", image);
        cv::waitKey(700);
    }


    return 0;
}