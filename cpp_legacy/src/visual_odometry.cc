#include <string>
#include <opencv2/opencv.hpp>
#include "visual_odometry.h"
#include "ara_slam.h"
#include "localisation/localisation.h"
#include "features/features.h"
#include "epipolar_geometry/epipolar.h"

#define _DEBUG_ true
namespace ara_slam {	
	// using local = ara_slam::localisation;
	using features_t = std::tuple<std::vector<cv::KeyPoint>,cv::Mat>;

	VisualOdometry::VisualOdometry(cv::Mat intrinsic){
		_state = odometry_state::initialising;
		K = intrinsic;
	}

	/**
	 * The initialisation process involves the following steps
	 * - Compute pose of frame 1 to frame 2 or to frame N, the is further broken down into
	 * 		- Extracting and matching keypoints between the `reference_frame` and current `frame`
	 * 		- Computing the essential matrix and homography matrix between the frames, done
	 * 			in `local::recover_pose`
	 * - Triangulate the points in 3D space
	 * - Determine the depth scale
	 */
	bool VisualOdometry::initialise(cv::Mat frame){
		_initialise_counter += 1;		
		if( reference_frame.empty() || _initialise_counter > 75){
			reference_frame = frame.clone();
			_initialise_counter = 0;
			return false; 		
		}
		//extract orb keypoints
		features_t img1_feats = features::extract_features(reference_frame);
        features_t img2_feats = features::extract_features(frame);
		//find matches between the keypoints
		std::vector<cv::DMatch> matches = features::compare_features(
			std::get<1>(img1_feats), std::get<1>(img2_feats), feature_matcher::flann
		);

		//debug purposes only - drawing matches		
		if( _DEBUG_ ){
			cv::Mat outImg;
			cv::drawMatches(reference_frame, std::get<0>(img1_feats), frame, std::get<0>(img2_feats), matches, outImg);
			cv::imshow("Matches", outImg);
		}
		
		std::vector<cv::KeyPoint> kp_1, kp_2;
		std::vector<cv::KeyPoint> all_kp1 = std::get<0>(img1_feats);
		std::vector<cv::KeyPoint> all_kp2 = std::get<0>(img2_feats);
		//order the matches so they correspond between `kp_1` and `kp_2`
		std::for_each(matches.begin(), matches.end(), [&kp_1, &kp_2,all_kp1,all_kp2](const auto m) mutable{
			kp_1.push_back( all_kp1[m.queryIdx] );
			kp_2.push_back( all_kp2[m.trainIdx] );
		});		

		std::vector<cv::Point2f> p1, p2;
		cv::KeyPoint::convert(kp_1,p1);
		cv::KeyPoint::convert(kp_2,p2);
		
		if( p1.size() > 0 && p2.size() > 0 ){
			auto pose_essential = epipolar::recover_pose_essential(p1,p2,K);
			std::cout << "Essential rotation " << std::get<0>(pose_essential) << std::endl;
			std::cout << "*****" << std::endl;
			auto pose_homography = epipolar::recover_pose_homography(p1,p2,K);
		}
		// if( !std::get<0>(pose).empty() ){
		// 	//successfully recovered the camera pose 
		// 	//TODO: Post initialisation stuff, local map et al
		// 	return true;
		// }
		return false;
	}

	const odometry_state VisualOdometry::state(){
		return _state;
	}
}