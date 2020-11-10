use std::fs;
use std::path;

use crate::data::DataLoader;

use rand::seq::SliceRandom;
use opencv::{core, imgcodecs, highgui, prelude::*};

extern crate alphanumeric_sort;

/// Used to determine which of the 4 folders to load
pub enum Lighting{
	Daylight,
	Flashlight,
	Fluorescent,
	Lamps
}

pub struct NewTsukubaDataset {
	// pub local_url: String,
	// pub data_type: Lighting,

	left_images: Vec<String>,
	right_images: Vec<String>,
	camera_tracks: Vec<String>
}

impl NewTsukubaDataset {
	/***
	Creates an instance of `NewTsukubaDataset` and loads in the left and right
	images based on `data_type`
	 */
	pub fn new(local_url: &String, data_type: Lighting) -> NewTsukubaDataset {
		let light_type = lighting_to_str(&data_type);
		// locate the url to the specified folder based on the light source type
		let url = format!("{}/{}", local_url, light_type);

		let mut contents: Vec<path::PathBuf> = fs::read_dir(&url).expect("Error parsing dataset")
							.map(|x| x.unwrap().path())
							.filter(|x| x.is_dir())
							.collect::<Vec<path::PathBuf>>();
		contents.sort();

		assert_eq!(contents.len(), 2, "There should only be two subfolders: left & right here");

		let left_url = &format!("{}/{}/{}", local_url, light_type, "left");
		let right_url = &format!("{}/{}/{}", local_url, light_type, "right");
		let left_path = path::Path::new(left_url);
		let right_path = path::Path::new(right_url);
		
		println!("Left: {} \nRight: {}", left_path.display(), right_path.display());

		let left_images = fs::read_dir(&left_path).expect("Failure parsing left images");
		let mut left: Vec<String> = left_images.map(|x| x.unwrap().path().display().to_string())
												.collect::<Vec<String>>();
		alphanumeric_sort::sort_str_slice(&mut left);

		let right_images = fs::read_dir(&right_url).expect("Failure parsing right images");
		let mut right: Vec<String> = right_images.map(|x| x.unwrap().path().display().to_string())
													.collect::<Vec<String>>();
		alphanumeric_sort::sort_str_slice(&mut right);

		assert_eq!(right.len(), left.len(), "Stereo pairs must be the same length");

		let tracks = parse_camera_track(local_url);

		return NewTsukubaDataset { 
			// local_url: local_url.to_string(), data_type: data_type, 
			left_images: left, right_images: right, camera_tracks: tracks
		};
	}

	/// Read the pair of stereo images at the specified index `idx`
	pub fn read(&self, idx: usize) -> (core::Mat, core::Mat) {
		assert!((idx < self.length()), "{} should be between 0 and {}", idx, self.length());
		let read_flag = imgcodecs::IMREAD_COLOR;
		let left = imgcodecs::imread(&self.left_images[idx], read_flag)
								.expect("Error reading left image");
		let right = imgcodecs::imread(&self.right_images[idx], read_flag)
								.expect("Error reading right image");
		
		(left,right)
	}
}

impl super::DataLoader for NewTsukubaDataset {
	fn length(&self) -> usize {
		return self.right_images.len();
	}

	fn download_url(&self) -> String {
		return String::from("https://home.cvlab.cs.tsukuba.ac.jp/dataset");
	}
}

fn lighting_to_str(light_type: &Lighting) -> String {
	match light_type {
		Lighting::Daylight => "daylight".to_string(),
		Lighting::Flashlight => "flashlight".to_string(),
		Lighting::Fluorescent => "fluorescent".to_string(),
		Lighting::Lamps => "lamps".to_string()
	}
}

/**
	Ground truth position of the stereo camera 
	(relative to the middle point of the stereo camera's baseline). 
	1800 poses (one for each frame). Each line contain
	6 float values: X Y Z A B C. Where (X, Y, Z) is the 3D position of the
	camera nad (A, B, C) are the Euler angles (in degrees) that represent
	the camera orientation.
 */
fn parse_camera_track(local_url: &String) -> Vec<String> {
	let track_url = &format!("{}/camera_track.txt",local_url);
	let contents = fs::read_to_string(path::Path::new(track_url)).expect("camera_track.txt not found");
	return contents.lines().map(|l| l.to_owned()).collect::<Vec<String>>();
}