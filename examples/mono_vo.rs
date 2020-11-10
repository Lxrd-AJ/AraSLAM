extern crate ara_slam;

use ara_slam::data::new_tsukuba;
use ara_slam::data::DataLoader;
use ara_slam::features;

use opencv::{core, highgui};
use rand::Rng;

fn main() -> opencv::Result<()>{
	let dataset_url = String::from("./datasets/NewTsukubaStereoDataset");
	let dataset = new_tsukuba::NewTsukubaDataset::new(&dataset_url, new_tsukuba::Lighting::Daylight);

	// Show a sample left and right image from the dataset
	let rand_idx: usize = rand::thread_rng().gen_range(0, dataset.length());
	let (rand_left, rand_right) = dataset.read(rand_idx);
	let mut rand_img = core::Mat::default()?;
	let mut input_image: core::Vector<core::Mat> = core::Vector::new();
	input_image.push(rand_left);
	input_image.push(rand_right);
	core::hconcat(&input_image, &mut rand_img)?;
	highgui::named_window("Sample Image", highgui::WINDOW_GUI_NORMAL)?;
	highgui::imshow("Sample Image", &rand_img)?;

	let kps = features::detect_features(&rand_img, features::Detector::SURF);
	let kps_image = features::draw_keypoints(&rand_img, &kps);
	highgui::imshow("Sample Image", &kps_image)?;

	loop {
		if highgui::wait_key(10)? > 0 {
			break;
		}
	}

	Ok(())
}