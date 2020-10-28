#ifndef _CALIBRATOR_H_
#define _CALIBRATOR_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <tuple>
#include "./../ara_slam.h"

namespace ara_slam { namespace calibration {
    using detected_corners_t = std::tuple<
                std::vector<std::vector<cv::Point3f>>, //world corners
                std::vector<std::vector<cv::Point2f>> //detected corners
            >;

    struct intrinsic_t{
        cv::Mat cam_matrix = cv::Mat::eye(3,3,CV_64F); //3x3 floating point matrix
        cv::Mat distortion = cv::Mat::zeros(8,1,CV_64F);
    };

    struct extrinsic_t{
        cv::Mat rotation_matrix = cv::Mat::eye(3,3,CV_64F);
        cv::Mat translation_vec = cv::Mat::zeros(3,1,CV_64F);
    };

    struct camera_matrix_t{
        intrinsic_t intrinsic;
        extrinsic_t extrinsic;
    };

    enum class rotation_format {
        rotation_matrix,
        rotation_vector
    };

    struct stereo_t {
        camera_matrix_t left;
        camera_matrix_t right;
        cv::Mat left_right_rotation = cv::Mat::eye(3,3,CV_64F);
        cv::Mat left_right_translation = cv::Mat::zeros(3,1,CV_64F);
        cv::Mat fundamental = cv::Mat::eye(3,3,CV_64F);
        cv::Mat essential = cv::Mat::eye(3,3,CV_64F);
        cv::Mat left_rectification = cv::Mat::eye(3,3,CV_64F);
        cv::Mat right_rectification = cv::Mat::eye(3,3,CV_64F);
        cv::Mat left_projection = cv::Mat::zeros(3,4,CV_64F);
        cv::Mat right_projection = cv::Mat::zeros(3,4,CV_64F);
        cv::Mat disparity_depth = cv::Mat::zeros(4,4,CV_64F);
    };
    
    detected_corners_t detect_corners(std::vector<cv::Mat> images, cv::Size board_size, float square_size);

    camera_matrix_t calibrate_camera(std::vector<cv::Mat> images, cv::Size, float);
    stereo_t calibrate_stereo_camera(std::vector<std::tuple<cv::Mat,cv::Mat>> stereo_pairs, cv::Size board_size, float square_size);

    double calculate_projection_error(intrinsic_t);

    cv::Mat reproject_image(intrinsic_t, cv::Mat);
    std::tuple<cv::Mat,cv::Mat> stereo_rectify(std::tuple<cv::Mat,cv::Mat> stereo_pair, stereo_t stereo_params, bool cropped = false);

    // extrinsic_t get_camera_chessboard_pose(cv::Size board_size, float square_size, intrinsic_t camera_matrix, cv::Mat image, rotation_format format = rotation_format::rotation_vector);
    auto get_camera_chessboard_pose(cv::Size board_size, float square_size, intrinsic_t camera_matrix, cv::Mat image, rotation_format format = rotation_format::rotation_vector) -> extrinsic_t;
}}

#endif

//for each image, append to tgt_points and img_points (corners)
//calculate the camera intrinsic params from tgt and img points
//use the camera intrinsic params to reproject and image
    //add a func to help show the images side by side
//--utility funcs---   
//also calculate the projection errors from tgt and img points
//add a func to save the camera intrinsic params to a yaml file
//enumeration for the calibration patterns