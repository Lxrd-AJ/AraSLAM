extern crate kiss3d;
extern  crate nalgebra;

use nalgebra::{Vector3, UnitQuaternion};
use kiss3d::window::{Window, State};
use kiss3d::light::Light;
use kiss3d::scene::SceneNode;
use std::path;

struct AppState {
	obj: SceneNode,
	rot: UnitQuaternion<f32>
}

impl State for AppState {
	fn step(&mut self, _: &mut Window) {
		self.obj.prepend_to_local_rotation(&self.rot);

		// Move the object some more
		let t = nalgebra::Translation3::new(1.0, 0.0, 0.0);
		
		self.obj.prepend_to_local_translation(&t);
	}
}

pub fn render_cube() {
	let mut window = Window::new("AraSLAM: Odometry");
	let mut cam_object = window.add_cone(2.0, 4.0);
	let cam_obj_file = path::Path::new("./datasets/GOPRO.obj");
	let cam_scale = Vector3::new(1.0,1.0,1.0);
	let mut cam_object2 = window.add_obj(cam_obj_file, cam_obj_file, cam_scale);
	cam_object.set_color(0.0, 1.0, 0.0);
	cam_object2.set_color(1.0, 0.0, 0.0);

	window.set_light(Light::StickToCamera);

	let rot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);
	
	//different ways of rendering the window

	//-- 1:
	// let state = AppState { obj: cam_object2, rot: rot };
	// window.render_loop(state);

	//-- 2:
	// while window.render() {
	// 	cam_object2.prepend_to_local_rotation(&rot);
	// 	let t = nalgebra::Translation3::new(1.0, 0.0, 0.0);
	// 	cam_object2.prepend_to_local_translation(&t);
	// }
	
	//-- 3:
	// window.show();

}