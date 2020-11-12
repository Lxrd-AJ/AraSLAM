extern crate ara_slam;
extern crate opencv;
extern crate nalgebra as na;
extern crate rand;

use ara_slam::data::new_tsukuba;
use ara_slam::data::DataLoader;
use ara_slam::features;
use ara_slam::visualisation as viz;
use ara_slam::camera;

use opencv::{core, highgui, prelude::*};
use na::{MatrixMN, U1, U3, U8};
use rand::Rng;

use std::thread;

// type Matrix3x3 = MatrixMN<f32, U3, U3>;
// type Matrix1x8 = MatrixMN<f32, U1, U8>;
// type Matrix1x3 = MatrixMN<f32, U1, U3>;

fn main() -> opencv::Result<()>{
	let dataset_url = String::from("./datasets/NewTsukubaStereoDataset");
	let dataset = new_tsukuba::NewTsukubaDataset::new(&dataset_url, new_tsukuba::Lighting::Daylight);

	// Show a sample left and right image from the dataset
	let rand_idx: usize = rand::thread_rng().gen_range(0, dataset.length());
	let (rand_left, rand_right) = dataset.read(rand_idx);
	println!("Image size ({},{})", rand_left.rows(), rand_left.cols());
	let mut rand_img = core::Mat::default()?;
	let mut input_image: core::Vector<core::Mat> = core::Vector::new();
	input_image.push(rand_left);
	input_image.push(rand_right);
	core::hconcat(&input_image, &mut rand_img)?;
	highgui::named_window("Sample Image", highgui::WINDOW_GUI_NORMAL)?;
	highgui::imshow("Sample Image", &rand_img)?;

	//Example showing how to detect keypoints
	let kps = features::detect_features(&rand_img, features::Detector::SURF);
	let kps_image = features::draw_keypoints(&rand_img, &kps);
	highgui::imshow("Sample Image", &kps_image)?;

	//Example showing the camera parameters
	let camera = dataset.camera_params();
	let intrinsic_params = camera.intrinsic();
	println!("{:?}", intrinsic_params.internal_camera);

	let odometer = ara_slam::VisualOdometer::new();

	// winodws can only be created on the main thread in MacOS, therefore this call
	// will fail. 
	// There is a fix for it here https://github.com/sebcrozet/kiss3d/pull/253
	// thread::spawn(move || {
	// 	println!("Spawned a new thread to handle 3D visualisations");
	// 	print!("{:?}", camera.intrinsic().internal_camera);
	// 	viz::show_camera();
	// });
	
	// viz::show_camera();

	loop {
		if highgui::wait_key(10)? > 0 {
			highgui::destroy_all_windows();
			break;
		}
	}

	Ok(())
}