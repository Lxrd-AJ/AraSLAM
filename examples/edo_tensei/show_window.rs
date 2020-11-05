extern crate ara_slam;
use opencv::{
	prelude::*,
	core,
	highgui,
	videoio
};

pub fn main() {
	ara_slam::show_window().unwrap();
}