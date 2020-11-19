extern crate kiss3d;
extern  crate nalgebra;

mod test_show_window; //pub 

use nalgebra::{Vector3, UnitQuaternion, Translation3, MatrixMN, U1, U6};
use kiss3d::window::{Window, State};
use kiss3d::light::Light;
use kiss3d::scene::SceneNode;
use std::path;

type Matrix1x3 = MatrixMN<f32, U1, U6>;

pub fn show_camera() {
	test_show_window::render_cube();
}

pub fn render_camera(trans: &Matrix1x3, rot: &Matrix1x3) -> Window {
	let t = trans.translation3();
	let r = rot.unit_quaternion();
	let mut window = Window::new("AraSLAM: Odometry");
	let camera_obj_file = path::Path::new("./datasets/GOPRO.obj");//TODO: Find a standard way to reference the file
	let camera_scale = Vector3::new(1.0, 1.0, 1.0);
	let mut camera3d = window.add_obj(camera_obj_file, camera_obj_file, camera_scale);

	window.set_light(Light::StickToCamera);

	return window;
}

trait PoseRepresentable {
	fn translation3(&self) -> Translation3<f64>;
	fn unit_quaternion(&self) -> UnitQuaternion<f64>;
}

impl PoseRepresentable for Matrix1x3 {
	fn translation3(&self) -> Translation3<f64> {
		let x = self[(0,0)] as f64;
		let y = self[(0,1)] as f64;
		let z = self[(0,2)] as f64;
		Translation3::new(x, y, z)
	}

	fn unit_quaternion(&self) -> UnitQuaternion<f64> { 
		let alpha = self[(0,0)] as f64;
		let beta = self[(0,1)] as f64;
		let gamma = self[(0,2)] as f64;
		UnitQuaternion::from_euler_angles(alpha, beta, gamma)
	}
}
