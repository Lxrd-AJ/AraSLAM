use opencv::{core, xfeatures2d, features2d, prelude::*};

pub enum Detector {
	SURF,
	SIFT
}

type KeyPoints = opencv::types::VectorOfKeyPoint;
type Descriptor = core::Mat;
type KeyPointDescriptors = (KeyPoints, Descriptor);
type Matches = opencv::types::VectorOfDMatch;
type VectorOfMatches = opencv::types::VectorOfVectorOfDMatch;

#[allow(dead_code)]
pub fn detect_features(image: &core::Mat, _method: Detector) -> KeyPointDescriptors {
	// let mut detector = features2d::FAST(image, keypoints, threshold, nonmax_suppression);
	let mut detector = xfeatures2d::SURF::create(500_f64, 3, 4, true, false).unwrap();
	let mut kps = KeyPoints::new();
	let mut fts = core::Mat::default().unwrap();
	let mask = core::no_array().unwrap();//core::Mat::default().unwrap();
	// detector.detect(image, &mut kps);

	// let _x = detector.detect(image, &mut kps, &mask);
	let _x = detector.detect_and_compute(image, &mask, &mut kps, &mut fts, false);

	return (kps, fts);
}

/// Compares the two descriptors using a FLANN based matcher with `k=2` nearest
/// neighbours and uses a threshold of `0.7` to filter the k neighbours.
/// Returns at most `max_matches` matches or less.
pub fn detect_matches( desc1: &Descriptor, desc2: &Descriptor, max_matches: usize ) -> Matches {
	let matcher = features2d::FlannBasedMatcher::create().unwrap();
	// let index_params = opencv::flann::IndexParams::default().unwrap();
	// let search_params = opencv::flann::SearchParams::new(50, 0.1, true, true).unwrap();
	// let matcher = features2d::FlannBasedMatcher::new(&index_params, &search_params);

	//-- 1: Detect using a FLANN based matcher
	let mut matches = VectorOfMatches::new();
	let mask = core::Mat::default().unwrap();//core::no_array().unwrap();
	let _x = matcher.knn_train_match(desc1, desc2, &mut matches, 2, &mask, true);
	
	//-- 2: Filter the matches using Lowe's ratio test
	let mut good_matches = Matches::new();
	for kmatches in matches {
		let k1 = kmatches.get(0).unwrap();
		let k2 = kmatches.get(1).unwrap();

		if k1.distance < (0.7 * k2.distance) {
			good_matches.push(k1);
		}
	}

	//-- 3: Take only the amount `max_matches` features or less
	let num_feats: usize = if good_matches.len() > max_matches { max_matches } else { good_matches.len() };
	let mut vec_good = good_matches.to_vec();
	vec_good.sort_by(|x,y| x.distance.partial_cmp(&y.distance).unwrap());
	vec_good = vec_good[0..num_feats].to_vec();
	good_matches = Matches::from(vec_good);
	return good_matches;
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