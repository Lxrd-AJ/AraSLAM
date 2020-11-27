extern crate opencv;

use std::error::Error;

use crate::Pose;
use crate::camera::Intrinsic;
use opencv::{prelude::*,core, calib3d};

type Points = opencv::types::VectorOfPoint2f;

/// computes the pose of Camera 2 relative to Camera 1's coordinate system. 
/// `p1` represents points in Camera 1 and `p2` respresents points in Camera 2
pub fn relative_pose( p1: &Points, p2: &Points, k: Intrinsic) -> (Pose, Points, Points) {
	let mut inliers = core::Mat::default().unwrap();
	let essential = calib3d::find_essential_mat_matrix(
			&p1, &p2, &k.cv_mat_k(), 
			opencv::calib3d::RANSAC, 0.99, 1.0, 
			&mut inliers
		).unwrap();
	println!("Essential matrix:\n{:?}", essential.to_vec_2d::<f64>().unwrap());
	let mut rotation = core::Mat::default().unwrap(); //3x3 matrix
	let mut translation = core::Mat::default().unwrap(); //1x3 matrix
	let _x = calib3d::recover_pose_camera(&essential, &p1, &p2, &k.cv_mat_k(), 
		&mut rotation, &mut translation, &mut inliers);
	println!("{:?}\n{:?}", rotation.to_vec_2d::<f64>().unwrap(), translation.to_vec_2d::<f64>().unwrap());
	let pose = Pose::from(&rotation, &translation);

	// `inliers` is now a Nx1 matrix of 1s and 0s depending representing epipolar inlier points
	let mut inliers_p1 = Points::new();
	let mut inliers_p2 = Points::new();

	for idx in 0..inliers.rows() {
		// let is_inlier = if *inliers.at::<i8>(idx).unwrap() == (1 as i8) { true } else { false };
		let is_inlier = match *inliers.at::<u8>(idx).unwrap() {
			1 => true,
			_ => false
		};
		if is_inlier {
			inliers_p1.push( p1.to_vec()[idx as usize]);
			inliers_p2.push( p2.to_vec()[idx as usize]);
		}
	}
	
	return (pose, inliers_p1, inliers_p2);
}

fn recover_pose_essential( p1: &Points, p2: &Points, k: core::Mat) {
	unimplemented!();
}