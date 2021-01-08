extern crate ara_slam;
extern crate opencv;
extern crate nalgebra as na;
extern crate rand;

use ara_slam::{OdometryStatus, data::{MonocularDataLoader, new_tsukuba}, normalise};

use opencv::{highgui};
use na::{MatrixMN, U1, U3};

type Matrix1x3 = MatrixMN<f64, U1, U3>;


fn main() -> opencv::Result<()>{
	let dataset_url = String::from("./datasets/NewTsukubaStereoDataset");
	let mut dataset = new_tsukuba::NewTsukubaDataset::new(&dataset_url, new_tsukuba::Lighting::Daylight);
	
	let mut odometer = ara_slam::VisualOdometer::new();
	let amount_read = odometer.initialise(&mut dataset);
	assert_eq!(odometer.status, OdometryStatus::Initialised, "The odometer is not initialised");
	println!("Initialised odometry from the first {} frames", amount_read);

	let odometer_poses = odometer.poses();
	let gt_translations: Vec<Matrix1x3> = dataset.poses().into_iter().map(|p| *p.translation()).collect();
	let updated_odometer_poses = normalise(&odometer_poses, &gt_translations);
	odometer.update_poses(&updated_odometer_poses);
	println!("Updated {} odometer poses", updated_odometer_poses.len());
	
	// Estimate the camera pose for the next 15 views
	for idx in amount_read..(amount_read+15) {
		println!("Step index {:}:", idx);
		let image = dataset.read(idx);
		let prev_image = dataset.read(idx-1);
		odometer.step(&image, &prev_image,idx);
		let _x = highgui::wait_key(20);
	}
	let amount_read = amount_read + 15;
	println!("{}", amount_read);

	loop {
		if highgui::wait_key(10)? > 0 {
			let _x = highgui::destroy_all_windows();
			break;
		}
	}

	Ok(())
}



















// Show a sample left and right image from the dataset
// use ara_slam::data::StereoDataLoader;
// let rand_idx: usize = rand::thread_rng().gen_range(0, dataset.length());
// let (rand_left, rand_right) = dataset.read(rand_idx);

// println!("Image size ({}, {})", rand_left.rows(), rand_left.cols());
// let mut rand_img = core::Mat::default()?;
// let mut input_image: core::Vector<core::Mat> = core::Vector::new();
// input_image.push(rand_left);
// input_image.push(rand_right);
// core::hconcat(&input_image, &mut rand_img)?;
// highgui::named_window("Sample Image", highgui::WINDOW_GUI_NORMAL)?;
// highgui::imshow("Sample Image", &rand_img)?;

// //Example showing how to detect keypoints
// let (kps, desc) = features::detect_features(&rand_img, features::Detector::SURF);
// let kps_image = features::draw_keypoints(&rand_img, &kps);
// highgui::imshow("Sample Image", &kps_image)?;

// //Example showing the camera parameters
// let camera = dataset.camera_params();
// let intrinsic_params = camera.intrinsic();
// println!("{:?}", intrinsic_params.internal_camera);


// winodws can only be created on the main thread in MacOS, therefore this call
	// will fail. 
	// There is a fix for it here https://github.com/sebcrozet/kiss3d/pull/253
	// thread::spawn(move || {
	// 	println!("Spawned a new thread to handle 3D visualisations");
	// 	print!("{:?}", camera.intrinsic().internal_camera);
	// 	viz::show_camera();
	// });
	

	// viz::show_camera();