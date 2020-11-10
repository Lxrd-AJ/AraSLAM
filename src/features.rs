use opencv::{core, xfeatures2d, features2d, prelude::*}; //features2d

pub enum Detector {
	SURF,
	SIFT
}

type KeyPoints = opencv::types::VectorOfKeyPoint;

#[allow(dead_code)]
pub fn detect_features(image: &core::Mat, _method: Detector) -> KeyPoints {
	// let mut detector = features2d::FAST(image, keypoints, threshold, nonmax_suppression);
	let mut detector = xfeatures2d::SURF::create(100.0,4,3,false,false).unwrap();
	let mut kps = KeyPoints::new();
	let mask = core::Mat::default().unwrap();
	// detector.detect(image, &mut kps);

	let _x = detector.detect(image, &mut kps, &mask);

	return kps;
}

/// Returns an image that contains the keypoints derived from `image` 
/// rendered into it
pub fn draw_keypoints(image: &core::Mat, keypoints: &KeyPoints) -> core::Mat {
	let mut result = core::Mat::default().unwrap();
	let color = core::Scalar::all(-1f64);
	let flag = features2d::DrawMatchesFlags::DEFAULT;
	let _x = features2d::draw_keypoints(image, keypoints, &mut result, color, flag);

	return result;
}