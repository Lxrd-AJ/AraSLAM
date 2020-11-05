extern crate ara_slam;

mod common;

#[test]
fn test_add() {
	common::hello_world();
	assert_eq!(ara_slam::add(3, 3), 6);
}

#[test]
fn test_kitti_dataset() {
	assert_eq!(ara_slam::data::kitti::hello_world(), 5);
}

#[test]
fn show_window() {
	let _res = ara_slam::show_window();
	assert_eq!(true, true);
}

#[test]
fn test_new_tsukuba() {
	let e = ara_slam::data::new_tsukuba::Lighting::Daylight;
	assert_eq!(1,1);
}