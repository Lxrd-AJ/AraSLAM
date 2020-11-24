extern crate opencv;

use crate::Pose;
use crate::camera::Intrinsic;
use opencv::{prelude::*,core, calib3d};

type Points = opencv::types::VectorOfPoint2f;

pub fn relative_pose( p1: &Points, p2: &Points, k: Intrinsic) -> Pose {
	let essential = calib3d::find_essential_mat_matrix(
			&p1, &p2, &k.cv_mat(), 
			opencv::calib3d::RANSAC, 0.99, 1.0, 
			&mut core::no_array().unwrap()
		);
	
	unimplemented!();
}

fn recover_pose_essential( p1: &Points, p2: &Points, k: core::Mat) {
	unimplemented!();
}