#ifndef _VISUAL_ODOMETRY_H
#define _VISUAL_ODOMETRY_H

#include <string>
#include <opencv2/opencv.hpp>
#include "ara_slam.h"
#include "localisation/localisation.h"
#include "features/features.h"


namespace ara_slam {
	// using as = ara_slam;

	class VisualOdometry{
	public:
		VisualOdometry(cv::Mat intrinsic);

		/**
		 * Estimate the pose of the `reference_frame` to the next frame. If it fails on the
		 * next frame, it returns `false` and the function should be called with 
		 * the next frame consecutively until it returns `true` when it has found a good
		 * pose with decent transformations.
		 * The function also changes the odometry state to `tracking` 
		 * 
		 */
		bool initialise(cv::Mat);
		const odometry_state state();
	private:
		int _initialise_counter; //used to ensure initialisation is completed within 1000 frames
		cv::Mat reference_frame;
		cv::Mat K; //intrinsic parameters of the camera
		odometry_state _state;
		std::vector<cv::Point3f> point_map; //all the triangulated 3D points are stored here
	};
}

#endif
