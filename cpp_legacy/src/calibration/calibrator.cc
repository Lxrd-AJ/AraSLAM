#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "calibrator.h"
#include "./../ara_slam.h"

namespace ara_slam { namespace calibration {
    cv::Mat dummy_func(int a){
        return cv::Mat(cv::Size(a,a), CV_8UC3, cv::Scalar(0,0,0));
    }

    std::vector<cv::Point3f> calculate_target_corners(cv::Size board_size, float square_size){
        //width = row i.e 8, height = cols i.e 6
        // Eigen::MatrixXd corners = Eigen::MatrixXd::Zero(board_size.width * board_size.height, 3);
        // corners.col(0) = Eigen::ArrayXd::LinSpaced(board_size.width, 0, board_size.width * square_size).replicate(board_size.height,0);
        std::vector<cv::Point3f> corners;
        for(int col=0; col < board_size.height; col++)
            for(int row=0; row < board_size.width; ++row)
                corners.push_back( cv::Point3f(row*square_size,col*square_size,0) );
        
        return corners;
    }

    detected_corners_t detect_corners(std::vector<cv::Mat> images, cv::Size board_size, float square_size){
        std::vector<std::vector<cv::Point2f>> detected_corners;
        std::vector<std::vector<cv::Point3f>> corners;
        int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        cv::TermCriteria term_criteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
        cv::Size image_size = images.front().size();

        for(auto const& image: images){
            std::vector<cv::Point2f> c_corners;
            cv::Mat img_clone = image.clone();
            cv::Mat gray_image;
            bool c_found = cv::findChessboardCorners(image, board_size, c_corners, flags);
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray_image, c_corners, cv::Size(11,11), cv::Size(-1,-1), term_criteria);
            cv::drawChessboardCorners(img_clone, board_size, cv::Mat(c_corners), c_found);
            
            detected_corners.push_back( c_corners );
            corners.push_back( calculate_target_corners(board_size, square_size) );
            // cv::imshow("Chessboard corners", img_clone);
            // cv::waitKey(200);
        }

        return std::make_tuple(corners, detected_corners);
    }

    stereo_t calibrate_stereo_camera(std::vector<std::tuple<cv::Mat,cv::Mat>> stereo_pairs, cv::Size board_size, float square_size){
        stereo_t stereo_data;
        std::vector<cv::Mat> left_images;
        std::vector<cv::Mat> right_images;
        
        int flags = cv::CALIB_FIX_INTRINSIC;

        for(auto const& stereo_pair: stereo_pairs){
            left_images.push_back( std::get<0>(stereo_pair) );
            right_images.push_back( std::get<1>(stereo_pair) );
        }
        cv::Size image_size = left_images.front().size();
        cv::Mat stereo_pair_errors;

        //Optimisation step 
        //Due to the high dimensionality of the parameter space and noise in the input data, the function can diverge from the correct solution. 
        //The intrinsic parameters can be estimated with high accuracy for each of the cameras individually (for example, using calibrateCamera ), you are recommended to do so 
        //then pass CALIB_FIX_INTRINSIC flag to the function along with the computed intrinsic parameters.
        stereo_data.left = calibrate_camera( left_images, board_size, square_size );
        stereo_data.right = calibrate_camera( right_images, board_size, square_size );
        
        detected_corners_t l_detected_corners = detect_corners( left_images, board_size, square_size );
        detected_corners_t r_detected_corners = detect_corners( right_images, board_size, square_size );
        std::vector<std::vector<cv::Point3f>> pattern_corners = std::get<0>(l_detected_corners);
        
        double final_error = cv::stereoCalibrate( pattern_corners, 
                                std::get<1>(l_detected_corners), std::get<1>(r_detected_corners), 
                                //refine the estimates of the intrinsic parameters of both the left and right
                                stereo_data.left.intrinsic.cam_matrix,
                                stereo_data.left.intrinsic.distortion, stereo_data.right.intrinsic.cam_matrix,
                                stereo_data.right.intrinsic.distortion, 
                                image_size, stereo_data.left_right_rotation, 
                                stereo_data.left_right_translation, stereo_data.essential, 
                                stereo_data.fundamental, stereo_pair_errors, flags 
                        );
        std::cout << "Final re-projection error value = " << final_error << std::endl;

        return stereo_data;
    }

    std::tuple<cv::Mat,cv::Mat> stereo_rectify(std::tuple<cv::Mat,cv::Mat> stereo_pair, stereo_t stereo_params, bool cropped){        
        cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;
        cv::Mat undistort_left, undistort_right;
        cv::Rect left_roi, right_roi;
        int flags = cv::CALIB_ZERO_DISPARITY;

        cv::stereoRectify(
                    stereo_params.left.intrinsic.cam_matrix, stereo_params.left.intrinsic.distortion,
                    stereo_params.right.intrinsic.cam_matrix, stereo_params.right.intrinsic.distortion,
                    std::get<0>(stereo_pair).size(), 
                    stereo_params.left_right_rotation, stereo_params.left_right_translation,
                    stereo_params.left_rectification, stereo_params.right_rectification,
                    stereo_params.left_projection, stereo_params.right_projection,
                    stereo_params.disparity_depth, 
                    flags, -1, cv::Size(0,0), &left_roi, &right_roi
        );
        cv::initUndistortRectifyMap(
                    stereo_params.left.intrinsic.cam_matrix, stereo_params.left.intrinsic.distortion,
                    stereo_params.left_rectification, stereo_params.left_projection, std::get<0>(stereo_pair).size(),
                    CV_32F, left_map_x, left_map_y
        );
        cv::initUndistortRectifyMap(
                    stereo_params.right.intrinsic.cam_matrix, stereo_params.right.intrinsic.distortion,
                    stereo_params.right_rectification, stereo_params.right_projection, std::get<1>(stereo_pair).size(),
                    CV_32F, right_map_x, right_map_y
        );

        cv::remap(std::get<0>(stereo_pair), undistort_left, left_map_x, left_map_y, cv::INTER_LINEAR );
        cv::remap(std::get<1>(stereo_pair), undistort_right, right_map_x, right_map_y, cv::INTER_LINEAR );

        if( cropped ){
            undistort_left = undistort_left(left_roi);
            undistort_right = undistort_right(right_roi);
        }

        return std::make_tuple(undistort_left, undistort_right);
    }

    //TODO: Refactor this function to use `detect_corners` instead
    camera_matrix_t calibrate_camera(std::vector<cv::Mat> images, cv::Size board_size, float square_size){
        // intrinsic_t cam_mtx { 0 };
        camera_matrix_t cam_mtx;
        std::vector<std::vector<cv::Point2f>> detected_corners;
        std::vector<std::vector<cv::Point3f>> corners;
        std::vector<cv::Mat> board_rotations;
        std::vector<cv::Mat> board_translations;
        int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        cv::TermCriteria term_criteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
        cv::Size image_size = images.front().size();

        int counter = 0;
        for(auto const& image: images){
            std::vector<cv::Point2f> c_corners;
            cv::Mat img_clone = image.clone();
            cv::Mat gray_image;
            bool c_found = cv::findChessboardCorners(image, board_size, c_corners, flags);
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray_image, c_corners, cv::Size(11,11), cv::Size(-1,-1), term_criteria);
            cv::drawChessboardCorners(img_clone, board_size, cv::Mat(c_corners), c_found);
            detected_corners.push_back( c_corners );
            corners.push_back( calculate_target_corners(board_size, square_size) );
            
            // cv::imshow("Input image", image);
            // cv::waitKey(1000);
            // std::string path = "./.tmp/chessboard_patterns_" + std::to_string(counter) + ".jpg";
            // cv::imwrite(path, img_clone);
            // ++counter;
        }

        double error = cv::calibrateCamera(corners, detected_corners, image_size, 
                                cam_mtx.intrinsic.cam_matrix, cam_mtx.intrinsic.distortion,
                                board_rotations, board_translations);
        std::cout << "Root-Mean-Square re-projection error = " << error << std::endl;

        return cam_mtx;
    }

    double calculate_projection_error(intrinsic_t cam_mtx){
        return -1.0; //TODO: Implement
    }

    cv::Mat reproject_image(intrinsic_t intrinsic, cv::Mat image){
        cv::Mat result;
        cv::undistort(image, result, intrinsic.cam_matrix, intrinsic.distortion);
        return result;
    }

    auto get_camera_chessboard_pose(cv::Size board_size, float square_size, intrinsic_t camera_matrix, cv::Mat image, rotation_format format) -> extrinsic_t {
        extrinsic_t pose;
        std::vector<cv::Point3f> board_points = calculate_target_corners(board_size, square_size);
        std::vector<cv::Point2f> chess_corners;
        int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        bool c_found = cv::findChessboardCorners(image, board_size, chess_corners, flags);        
        if( !c_found ) //TODO: Determine what to do if the chess corners are not found
            std::cout << "FAILED TO FIND CHESSBOARD CORNERS" << std::endl;
        else{            
            cv::Mat rotation = cv::Mat::zeros(3,1,CV_64F);
            cv::solvePnPRansac(board_points, chess_corners, camera_matrix.cam_matrix, camera_matrix.distortion, rotation, pose.translation_vec, false );
            if( format == rotation_format::rotation_matrix)
                cv::Rodrigues(rotation, pose.rotation_matrix);
            else
                pose.rotation_matrix = rotation;
        }
        return pose;
    }
}}