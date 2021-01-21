extern crate opencv;

use std::error::Error;

use crate::{Pose, point_pairs};
use crate::camera::Intrinsic;
use crate::features::{detect_features, detect_matches};
use opencv::{prelude::*,core, calib3d, sfm};

type Points = opencv::types::VectorOfPoint2f;
type Points3D = opencv::types::VectorOfPoint3d;
type KeyPoints = opencv::types::VectorOfKeyPoint;
type Descriptor = core::Mat;
type KeyPointDescriptors = (KeyPoints, Descriptor);
type Matches = opencv::types::VectorOfDMatch;

/// computes the pose of Camera 2 relative to Camera 1's coordinate system. 
/// `p1` represents points in Camera 1 and `p2` respresents points in Camera 2
pub fn relative_pose( p1: &Points, p2: &Points, k: Intrinsic) -> (Pose, Points, Points) {
	let mut inliers = core::Mat::default().unwrap();
	let essential = calib3d::find_essential_mat_matrix(
			&p1, &p2, &k.cv_mat_k(), 
			opencv::calib3d::RANSAC, 0.99, 1.0, 
			&mut inliers
		).unwrap();
	
	let mut rotation = core::Mat::default().unwrap(); //3x3 matrix
	let mut translation = core::Mat::default().unwrap(); //1x3 matrix
	let _x = calib3d::recover_pose_camera(&essential, &p1, &p2, &k.cv_mat_k(), 
		&mut rotation, &mut translation, &mut inliers);
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

/// Eliminates outliers from the feature matches between `p1` and `p2` using the epipolar constraint
pub fn filter_epipolar_inliers(p1: &Points, p2: &Points, k: Intrinsic) -> (Points, Points) {
	let (_pose, points1, points2) = relative_pose(p1, p2, k);
	return (points1, points2);
}

/// Given a triplet of `KeyPointDescriptors` (points and descriptor pairs) from 3 consecutive images
/// return a triplet of points (p1,p2,p3) that are visible in all 3 images
/// where p1[1] is the same point in p2[1] and p3[1]
pub fn intersect(kpd1: KeyPointDescriptors, kpd2: KeyPointDescriptors, kpd3: KeyPointDescriptors, k: Intrinsic) 
	-> (Points, Points, Points) {
	
	let (kp1, d1) = kpd1;
	let (kp2, d2) = kpd2;
	let (kp3, d3) = kpd3;

	let m1_2 = detect_matches(&d1, &d2, 150);
	let m2_3 = detect_matches(&d2, &d3, 150);

	let pointpairs_1_2 = index_pairs(&m1_2);
	let pointpairs_2_3 = index_pairs(&m2_3);
	
	let pointpairs_1_2 = filter_inliers(&pointpairs_1_2, &kp1, &kp2, &k);
	println!("=> Inliers P1 to P2: {:} ", pointpairs_1_2.len());
	let pointpairs_2_3 = filter_inliers(&pointpairs_2_3, &kp2, &kp3, &k);
	println!("=> Inliers P2 to P3: {:} ", pointpairs_2_3.len());
	let point_triplet = filter_index_pairs(&pointpairs_1_2, &pointpairs_2_3);
	
	let p1: Points = point_triplet.iter().map(|t| kp1.to_vec()[t.0 as usize].pt).collect();
	let p2: Points = point_triplet.iter().map(|t| kp2.to_vec()[t.1 as usize].pt).collect();
	let p3: Points = point_triplet.iter().map(|t| kp3.to_vec()[t.2 as usize].pt).collect();

	(p1, p2, p3)
}

/// returns the 3D position of the 2D points in `p1` and `p2`
/// Done using [cv::sfm::triangulatePoints](https://docs.opencv.org/4.5.1/d0/dbd/group__triangulation.html#ga211c855276b3084f3bbd8b2d9161dc74)
pub fn triangulate(p1: &Points, p2: &Points, pose1: &Pose, pose2: &Pose, k: &Intrinsic) -> Points3D {
	let mut points2d = opencv::types::VectorOfVectorOfPoint2f::new();
	points2d.push(p1.clone());
	points2d.push(p2.clone());

	let mut projection_matrices = opencv::types::VectorOfMat::new();
	let mut proj1 = core::Mat::default().unwrap();
	let _x = sfm::projection_from_k_rt(&k.cv_mat_k(), &pose1.r_asmat(), &pose1.t_asmat(), &mut proj1);
	let mut proj2 = core::Mat::default().unwrap();
	let _x = sfm::projection_from_k_rt(&k.cv_mat_k(), &pose2.r_asmat(), &pose2.t_asmat(), &mut proj2);
	projection_matrices.push(proj1);
	projection_matrices.push(proj2);

	let mut points3d = Points3D::new();
	let _x = sfm::triangulate_points(&points2d, &projection_matrices, &mut points3d);

	return points3d;
}

fn filter_inliers(pair: &Vec<(i32,i32)>, kp1: &KeyPoints, kp2: &KeyPoints, k: &Intrinsic) -> Vec<(i32,i32)>{
	let p1: Points = pair.iter().map(|p| kp1.to_vec()[p.0 as usize].pt).collect();
	let p2: Points = pair.iter().map(|p| kp2.to_vec()[p.1 as usize].pt).collect();

	let mut inliers = core::Mat::default().unwrap();
	let essential = calib3d::find_essential_mat_matrix(
			&p1, &p2, &k.cv_mat_k(), 
			opencv::calib3d::RANSAC, 0.99, 2.0, 
			&mut inliers
		).unwrap();
	
	let mut rotation = core::Mat::default().unwrap(); //3x3 matrix
	let mut translation = core::Mat::default().unwrap(); //1x3 matrix
	let num_inliers = calib3d::recover_pose_camera(&essential, &p1, &p2, &k.cv_mat_k(), 
	&mut rotation, &mut translation, &mut inliers).unwrap();

	assert_eq!(inliers.rows() as usize, pair.len(), "The number of index pairs and inliers must match");
	println!("*********\nNum inliers: {:}", num_inliers);
	// println!("Calibration data:\n{:}", k.internal_camera);
	// let mat = k.cv_mat_k();
	// fn print(mat: &core::Mat){
	// 	for row in 0..mat.rows() {
	// 		print!("| ");
	// 		for col in 0..mat.cols() {
	// 			print!("{:}\t", *mat.at_2d::<f64>(row, col).unwrap());
	// 		}
	// 		println!("|");
	// 	}
	// }
	// println!("Intrinsic matrix");
	// print(&mat);
	// println!("");
	// print(&translation);
	
	let mut result: Vec<(i32,i32)> = Vec::new();

	for idx in 0..inliers.rows() {
		let is_inlier = match *inliers.at::<u8>(idx).unwrap() {
			1 => true,
			_ => false
		};
		if is_inlier {
			result.push(pair[idx as usize]);
		}
	}

	return result;
}

fn index_pairs(matches: &Matches) -> Vec<(i32,i32)> {
	matches.to_vec().iter().map(|m| (m.query_idx, m.train_idx)).collect()
}

fn filter_index_pairs(pair1: &Vec<(i32,i32)>, pair2: &Vec<(i32,i32)>) -> Vec<(i32,i32,i32)> {
	let mut point_triplet: Vec<(i32,i32,i32)> = Vec::new();

	for i in 0..pair1.len() {
		for j in 0..pair2.len() {
			let (a1, a2) = pair1[i];
			let (b1, b2) = pair2[j];

			if a2 == b1 {
				point_triplet.push((a1,a2,b2));
				break;
			}
		}
	}

	point_triplet.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap() );
	return point_triplet;
}