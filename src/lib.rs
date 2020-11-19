extern crate opencv;
extern crate nalgebra as na;

use std::collections::HashMap;

use opencv::{
	prelude::*,
	core,
	highgui,
	videoio
};
use na::{MatrixMN, U1, U6};

pub mod data;
pub mod features;
pub mod visualisation;
pub mod camera;


type KeyPoints = opencv::types::VectorOfKeyPoint;
type Descriptor = core::Mat;
type Matches = opencv::types::VectorOfDMatch;
type Matrix1x3 = MatrixMN<f32, U1, U6>;
type StereoPair = (core::Mat, core::Mat);
type FrameID = i32;

pub enum OdometryStatus {
	Limbo,
	Initialising,
	Tracking,
	Lost
}

#[derive(Clone)]
struct Pose {
	rotation: Matrix1x3,
	translation: Matrix1x3
}

impl Pose {
	pub fn new() -> Pose {
		Pose {
			rotation: Matrix1x3::zeros(),
			translation: Matrix1x3::zeros()
		}
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
	matches: Matches //the matches between the prev and current image
}

pub struct VisualOdometer {
	frames: Vec<Frame>,
	connections: HashMap<(FrameID, FrameID), Connection>, // (prev, cur) represents the keys
	status: OdometryStatus,
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

	// Initialises the odometer and returns the total number of frames read
	/// Calculates the relative pose between the first frame and the second frame
	/// If the calculation fails, then the next image or the one after that is used until
	/// a successful pose is calculated.
	pub fn initialise(&mut self, using: &mut impl data::DataLoader<Item=StereoPair> ) -> usize{
		let dataset = using;
		

		//-- 1: Compute the features of the first image and store the information about the image as the first frame
		let (first_image, _) = dataset.next().unwrap();
		let (kps1, desc1) = features::detect_features(&first_image, features::Detector::SURF);
		let first_frame = Frame { id: 0, pose: Pose::new(), points: kps1.clone(), descriptors: desc1.clone() };
		let first_frame_c = first_frame.clone();

		self.frames.push(first_frame);
		self.status = OdometryStatus::Initialising;
		let mut amount_read = 1;

		//-- render the camera
		let mut window = visualisation::render_camera(&first_frame_c.pose.translation, &first_frame_c.pose.rotation);

		for (idx, (left, _)) in dataset.enumerate() {
			// -- 2: Match features between the current image and the first image
			let (kpsn, descn) = features::detect_features(&left, features::Detector::SURF);
			let matches = features::detect_matches(&desc1, &descn);
			amount_read += 1;
			println!("Dataset index {}", idx);

			//-- Draw the matches
			let mut drawn_matches = core::Mat::default().unwrap();
			let color = core::Scalar::all(-1f64);
			let mask = opencv::types::VectorOfi8::new();
			let flag = opencv::features2d::DrawMatchesFlags::DEFAULT;
			let _x = opencv::features2d::draw_matches(&first_image, &kps1, &left, &kpsn, &matches, &mut drawn_matches,
				color, color, &mask, flag);
			highgui::named_window("MATCHES", highgui::WINDOW_GUI_NORMAL).unwrap();
			highgui::imshow("MATCHES", &drawn_matches);
			highgui::wait_key(10);

			//-- 3: Compute pose relative to first frame

			if amount_read > 20 {
				break;
			}

			window.render();
		}

		amount_read
	}

	// pub fn initialise(&self, using: &mut impl data::StereoDataLoader<core::Mat, Item=StereoPair> ) -> usize{
	// 	let dataset = using;
	// 	let mut amount_read = 0;

	// 	let (l,r) = dataset.next().unwrap();

	// 	for img in dataset {
	// 		amount_read += 1;
	// 	}

	// 	amount_read
	// }
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
