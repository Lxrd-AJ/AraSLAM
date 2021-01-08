extern crate opencv;
extern crate nalgebra as na;
extern crate itertools;

use std::collections::HashMap;

use opencv::{
	prelude::*,
	core,
	highgui,
	videoio
};
use na::{MatrixMN, U1, U3};
use itertools::izip;

pub mod data;
pub mod features;
pub mod visualisation;
pub mod camera;
mod geometry;


type KeyPoints = opencv::types::VectorOfKeyPoint;
type Points = opencv::types::VectorOfPoint2f;
type Descriptor = core::Mat;
type Matches = opencv::types::VectorOfDMatch;
type Matrix1x3 = MatrixMN<f64, U1, U3>;
type Matrix3x3 = MatrixMN<f64, U3, U3>;
// type StereoPair = (core::Mat, core::Mat);
type FrameID = i32;

#[derive(Clone, Debug, PartialEq)]
pub enum OdometryStatus {
	Limbo,
	Initialising,
	Initialised,
	Tracking,
	Lost
}

#[derive(Clone, Debug, Copy)]
pub struct Pose {
	rotation: Matrix3x3,
	translation: Matrix1x3
}

impl Pose {
	pub fn new() -> Pose {
		Pose {
			rotation: Matrix3x3::identity(),
			translation: Matrix1x3::zeros()
		}
	}

	/// computes the pose struct from the rotation and translation matrix supplied
	/// 
	/// # Arguments
	/// - `r`: a 3x3 rotation matrix
	/// - `t`: a 1x3 translation vector
	pub fn from(r: &core::Mat, t: &core::Mat) -> Pose {
		let mut tv = Matrix1x3::zeros();
		tv[(0,0)] = *t.at::<f64>(0).unwrap();
		tv[(0,1)] = *t.at::<f64>(1).unwrap();
		tv[(0,2)] = *t.at::<f64>(2).unwrap();

		let mut rm = Matrix3x3::zeros();
		rm[(0,0)] = *r.at_2d::<f64>(0,0).unwrap();
		rm[(0,1)] = *r.at_2d::<f64>(0,1).unwrap();
		rm[(0,2)] = *r.at_2d::<f64>(0,2).unwrap();
		rm[(1,0)] = *r.at_2d::<f64>(1,0).unwrap();
		rm[(1,1)] = *r.at_2d::<f64>(1,1).unwrap();
		rm[(1,2)] = *r.at_2d::<f64>(1,2).unwrap();
		rm[(2,0)] = *r.at_2d::<f64>(2,0).unwrap();
		rm[(2,1)] = *r.at_2d::<f64>(2,1).unwrap();
		rm[(2,2)] = *r.at_2d::<f64>(2,2).unwrap();

		Pose {
			rotation: rm,
			translation: tv
		}
	}

	pub fn rotation(&self) -> &Matrix3x3 {
		&self.rotation
	}

	pub fn translation(&self) -> &Matrix1x3 {
		&self.translation
	}
}

#[derive(Clone)]
struct Frame {
	id: FrameID,
	pose: Pose,
	points: KeyPoints,
	descriptors: Descriptor
}

/// Handles connections between two frames `Frame`
struct Connection {
	points1: Points,
	points2: Points
}

pub struct VisualOdometer {
	frames: Vec<Frame>,
	connections: HashMap<(FrameID, FrameID), Connection>, // (prev, cur) represents the keys
	pub status: OdometryStatus,
	reference_frame: core::Mat,
	camera_params: camera::Camera,
	camera_tracks: Vec<i32> //TODO: Replace i32 with the 1x6 matrix representing camera translation and rot
}

impl VisualOdometer {

	pub fn new() -> VisualOdometer {
		VisualOdometer {
			frames: Vec::new(),
			connections: HashMap::new(),
			status: OdometryStatus::Limbo,
			reference_frame: core::Mat::default().unwrap(),
			camera_params: camera::Camera::new(),
			camera_tracks: Vec::new()
		}
	}

	/// Initialises the odometer and returns the total number of frames read
	/// Calculates the relative pose between the first frame and the second frame
	/// If the calculation fails, then the next image or the one after that is used until
	/// a successful pose is calculated.
	pub fn initialise(&mut self, using: &mut impl data::MonocularDataLoader<core::Mat> ) -> usize{
		let dataset = using;
		const MAX_READ: usize = 10;

		//-- 1: Compute the features of the first image and store the information about the image as the first frame
		let first_image = dataset.read(0);
		let (kps1, desc1) = features::detect_features(&first_image, features::Detector::SURF);
		let first_frame = Frame { id: 0, pose: Pose::new(), points: kps1.clone(), descriptors: desc1.clone() };

		self.frames.push(first_frame);
		self.status = OdometryStatus::Initialising;
		let mut amount_read = 1;

		for idx in 1..MAX_READ {
			let left = dataset.read(idx);
			// -- 2: Match features between the current image and the first image
			let (kpsn, descn) = features::detect_features(&left, features::Detector::SURF);
			let matches = features::detect_matches(&desc1, &descn, 150);
			amount_read += 1;

			//-- 3: Compute pose of the current frame relative to first frame using the matched features
			let (p1, pn) = point_pairs( &kps1, &kpsn, &matches ); 
			let (pose, p1, p2) = geometry::relative_pose(&p1, &pn, self.camera_params.intrinsic());
			let frame_n = Frame { id: (idx as i32), pose: pose, points: kpsn.clone(), descriptors: descn.clone() };
			self.frames.push(frame_n);

			//-- 4: Add a connection between these two frames and finish initialisation
			let conn = Connection{ points1: p1, points2: p2 };
			self.connections.insert((0,(idx as i32)), conn);


			// For now there is no efficient way to determine if the pose calculation failed.
			// so assume it worked correctly and exit initialisation given that we have found a pose relative
			// to the first frame
			self.status = OdometryStatus::Initialised;
			break;
		}

		amount_read
	}

	
	/// Steps through the visual odometry process using the new observation `img`
	/// NB: `prev_img` is used for debugging so far and can be safely removed
	pub fn step(&mut self, img: &core::Mat, prev_img: &core::Mat, step_idx: usize) {
		// match points between the previous image and this image
		let (cur_kps, cur_desc) = features::detect_features(img, features::Detector::SURF);
		let prev_frame: Frame = self.frames[step_idx-1].clone(); //self.frames.last().unwrap().clone(); 
		let prev_kps = prev_frame.points; 
		let prev_desc = prev_frame.descriptors;
		let matches = features::detect_matches(&prev_desc, &cur_desc, 150);
		
		//draw the matches
		highgui::named_window("MATCHES", highgui::WINDOW_GUI_NORMAL).unwrap();
		let x = draw_matches(prev_img, &prev_kps, img, &cur_kps, &matches);
		let _x = highgui::imshow("MATCHES", &x.unwrap());

		//eliminate outlier points from the matches
		// let (prev_pts, cur_pts) = point_pairs(&prev_kps, &cur_kps, &matches);
		// let (prev_pts, cur_pts) = geometry::filter_epipolar_inliers(&prev_pts, &cur_pts, self.camera_params.intrinsic());
		
		// Get the points from frame_{n-1} that are also in the current frame
		let prevprev_frame = self.frames[step_idx-2].clone();
		let prevprev_kps = prevprev_frame.points;
		let prevprev_desc = prevprev_frame.descriptors;
		// let matches_cur2prevprev = features::detect_matches(&prevprev_desc, &cur_desc, 150);
		// let (prevprev_pts, _cur_pts) = point_pairs(&prevprev_kps, &cur_kps, &matches_cur2prevprev);
		
		let point_triplet = geometry::intersect((prevprev_kps, prevprev_desc), 
				(prev_kps, prev_desc), (cur_kps, cur_desc), self.camera_params.intrinsic());


		// TODO: Determine the rotation and translation of the current view `img`. Using either
		//	- 2D to 2D correspondences: geometry::relative_pose
		//  - 3D to 2D correspondences:
		//		- using `point_triplet` i can determine 3D points from p1 & p2 and use that to do 3D to 2D (p3)
		
		// TODO: determine the 3D points in the real world using `point_triplet`
		//	- I can use [cv::sfm::triangulatePoints](https://docs.opencv.org/4.5.1/d0/dbd/group__triangulation.html#ga211c855276b3084f3bbd8b2d9161dc74)
		
	}

	/// Returns the pose for every frame seen by the odometer
	pub fn poses(&self) -> Vec<Pose> {
		self.frames.iter().map(|frame| frame.pose).collect()
	}

	pub fn update_poses(&mut self, poses: &Vec<Pose>) {
		assert_eq!(self.frames.len(), poses.len(), "Number of poses and frames have to be equal");
		for i in 0..poses.len() {
			self.frames[i].pose = poses[i];
		}
	}
}

/// Given the keypoints from each image and their point matches, the matches
/// are used to filter the keypoints and an orderded pair of points are 
/// returned
fn point_pairs(kp1: &KeyPoints, kp2: &KeyPoints, matches: &Matches) -> (Points, Points) {
	// let mut p1 = Points::new();
	// let mut p2 = Points::new();

	// for m in matches {
	// 	let point1 = kp1.to_vec()[(m.query_idx as usize)].pt;
	// 	let point2 = kp2.to_vec()[(m.train_idx as usize)].pt;
		
	// 	p1.push(point1);
	// 	p2.push(point2);
	// }

	let p1 = matches
		.to_vec().iter()
		.map(|m| 
			kp1.to_vec()[(m.query_idx as usize)].pt
		).collect::<Points>();

	let p2 = matches
		.to_vec().iter()
		.map(|m| kp2.to_vec()[(m.train_idx as usize)].pt).collect::<Points>();

	return (p1, p2);
}


/// Computes the scale factor from the translations magnitudes and uses that to normalise
/// the poses
pub fn normalise(poses: &Vec<Pose>, true_translations: &Vec<Matrix1x3>) -> Vec<Pose> {
	// move the cameras to the origins
	let first_pose = poses[0];
	let poses: Vec<Pose> = poses.iter()
				.map(|p| Pose { rotation: *p.rotation(), translation: p.translation() - first_pose.translation()})
				.collect();
	let gt_poses: Vec<Matrix1x3> = true_translations.iter().take(poses.len()).map(|x| *x).collect();
	let pose_magnitudes: Vec<f64> = poses.iter().map(|p| p.translation().norm() as f64).collect();// L2 norms
	let gt_magnitudes: Vec<f64> = gt_poses.iter().map(|gt| gt.norm() as f64).collect();
	let mut possible_scale_factors: Vec<f64> = Vec::new();
	for (x,y) in izip!(&gt_magnitudes[1..], &pose_magnitudes[1..]) {
		possible_scale_factors.push(x/y);
	}

	let scale_factor = possible_scale_factors.pop().unwrap(); //TODO: Use the median of the list instead
	let updated_poses: Vec<Pose> = poses.iter()
				.map(|p| 
					Pose { rotation: p.rotation() * first_pose.rotation(), translation: p.translation() * scale_factor})
				.collect();
	return updated_poses;
}

fn draw_matches(first_image: &core::Mat, kps1: &KeyPoints, second_image: &core::Mat, kps2: &KeyPoints, matches: &Matches) -> opencv::Result<core::Mat> {
	let mut drawn_matches = core::Mat::default()?;
	let color = core::Scalar::new(10_f64,10f64,255f64,1f64);
	let match_color = core::Scalar::new(10f64, 255f64, 10f64, 1f64);
	let mask = opencv::types::VectorOfi8::new();
	let flag = opencv::features2d::DrawMatchesFlags::DEFAULT;
	opencv::features2d::draw_matches(first_image, kps1, second_image, kps2, matches, &mut drawn_matches,
		match_color, color, &mask, flag)?;
	Ok(drawn_matches)
}


























































pub fn add(a:i32, b:i32) -> i32 {
	a + b
}

pub fn test_module_system() {
	data::kitti::hello_world();
}

pub fn test_new_tsukuba_ds() {
	let _e = data::new_tsukuba::Lighting::Daylight;
}

pub fn show_window() -> opencv::Result<()> {
	let window = "Video Capture";
	highgui::named_window(window,1)?;
	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
	let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    loop {
        let mut frame = core::Mat::default()?;
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
	}
	
	Ok(())
}


#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*; 
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
	}
	
	mod addition_tests {
		use super::*;

		#[test]
		fn test_add() {
			assert_eq!(add(4,5), 9);
		}
	}
}
