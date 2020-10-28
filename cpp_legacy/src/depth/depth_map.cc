#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/calib3d.hpp>
// #include <opencv2/ximgproc/disparity_filter.hpp>
#include "./../ara_slam.h"
#include "./../calibration/calibrator.h"

namespace ara_slam { namespace depth {

   
    cv::Mat compute_depth(cv::Mat l, cv::Mat r){
        cv::Mat disparity,filt_disp,l_disp,r_disp,lg,rg;
        cv::cvtColor( l,lg,cv::COLOR_BGR2GRAY );
        cv::cvtColor( r,rg,cv::COLOR_BGR2GRAY );
        const int block_size = 7; //5
        const int max_disparity = 16 * 1; //reduce for less black cols on left side
        //0,5
        cv::Ptr<cv::StereoSGBM> stereo_sgbm = cv::StereoSGBM::create(0,max_disparity,block_size);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(stereo_sgbm);
        stereo_sgbm->setP1(24*block_size*block_size);
        stereo_sgbm->setP2(96*block_size*block_size);
        stereo_sgbm->setPreFilterCap(63);
        stereo_sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        // stereo_sgbm->setSpeckleWindowSize(50);

        stereo_sgbm->compute(lg,rg,l_disp);
        right_matcher->compute(rg,lg,r_disp);
        // cv::normalize(l_disp, disparity, 0, 255, cv::NORM_MINMAX , CV_8U); 

        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls = cv::ximgproc::createDisparityWLSFilter(stereo_sgbm);
        wls->setLambda(4000);
        wls->setSigmaColor(1.5);
        wls->filter(l_disp,lg,filt_disp,r_disp);        
        
        cv::normalize(filt_disp, disparity, 0, 255, cv::NORM_MINMAX , CV_8U); 
        return disparity; 
    }

    std::vector<cv::Point3i> triangulate_points(std::vector<cv::Point3i> ps, calibration::stereo_t k_stereo){
        std::vector<cv::Point3i> points;
        Eigen::MatrixXf Q(4,4);
        cv::cv2eigen(k_stereo.disparity_depth, Q);   
        std::cout << k_stereo.left_projection << std::endl;
        std::cout << k_stereo.left_projection.inv() << std::endl;
                    
        for(const auto p: ps){
            Eigen::VectorXf vp(4);
            vp(0) = p.x; vp(1) =  p.y; vp(2) = p.z; vp(3) = 1;            
            Eigen::VectorXf pe = Q * vp;            
            pe /= pe(3);
            points.push_back( cv::Point3i(pe(0),pe(1),pe(2)) );            
        }
        return points;
    }

    std::vector<cv::Point3f> world_xyz_from_keypoints(std::vector<cv::KeyPoint> kps, cv::Mat depth_map){   
        std::vector<cv::Point3f> point_cloud;
        for(const auto kp: kps){            
            cv::Vec3f _3Dp = depth_map.at<cv::Vec3f>(kp.pt.y, kp.pt.x);            
            cv::Point3f p = cv::Point3f(_3Dp[0],_3Dp[1],_3Dp[2]);            
            point_cloud.push_back(p);            
        }
        return point_cloud;
    }

    std::vector<cv::Point2i> world_xyz_to_image(std::vector<cv::Point3f> pts, calibration::stereo_t stereo){
        std::vector<cv::Point2f> image_points;        
        cv::Mat R = stereo.left.extrinsic.rotation_matrix;
        cv::Mat t = stereo.left.extrinsic.translation_vec;
        cv::Mat K = stereo.left.intrinsic.cam_matrix;
        cv::Rodrigues(R,R);        

        cv::projectPoints(pts,R,t,K,std::vector<float>(),image_points);
        std::vector<cv::Point2i> result;
        cv::Mat(image_points).convertTo(result, cv::Mat(result).type());
        
        return result;
    }
}}