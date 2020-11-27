mod camera;

extern crate nalgebra as na;
extern crate opencv;

use na::{MatrixMN, U1, U3, U8};
use opencv::{prelude::*,core};

type Matrix3x3 = MatrixMN<f64, U3, U3>;
type Matrix1x8 = MatrixMN<f64, U1, U8>;
type Matrix1x3 = MatrixMN<f64, U1, U3>;

#[derive(Copy, Clone)]
pub struct Intrinsic {
	pub internal_camera: Matrix3x3, //K
	pub distortion: Matrix1x8
}

impl Intrinsic {
	pub fn new() -> Intrinsic {
		Intrinsic {
			internal_camera: Matrix3x3::identity(),
			distortion: Matrix1x8::zeros()
		}
	}

	pub fn from(focal_length: (f64, f64), principal_point: (f64, f64)) -> Intrinsic {
		let (fx, fy) = focal_length;
		let (px, py) = principal_point;
		let K = Matrix3x3::new(fx, 0.0, px,
												0.0, fy, py,
												0.0, 0.0, 1.0);
		
		Intrinsic {
			internal_camera: K,
			distortion: Matrix1x8::zeros()
		}
	}

	/// returns an OpenCV matrix representing the `internal_camera` intrinsic i.e `K`
	pub fn cv_mat_k(&self) -> core::Mat {
		let mut k: core::Mat = core::Mat::zeros(3, 3, f64::typ()).unwrap().to_mat().unwrap();
		*k.at_2d_mut::<f64>(0,0).unwrap() = self.internal_camera[(0,0)];
		*k.at_2d_mut::<f64>(0,1).unwrap() = self.internal_camera[(0,1)];
		*k.at_2d_mut::<f64>(0,2).unwrap() = self.internal_camera[(0,2)];

		*k.at_2d_mut::<f64>(1,0).unwrap() = self.internal_camera[(1,0)];
		*k.at_2d_mut::<f64>(1,1).unwrap() = self.internal_camera[(1,1)];
		*k.at_2d_mut::<f64>(1,2).unwrap() = self.internal_camera[(1,2)];

		*k.at_2d_mut::<f64>(2,0).unwrap() = self.internal_camera[(2,0)];
		*k.at_2d_mut::<f64>(2,1).unwrap() = self.internal_camera[(2,1)];
		*k.at_2d_mut::<f64>(2,2).unwrap() = self.internal_camera[(2,2)];

		k
	}
}


pub struct Extrinsic {
	pub rotation: Matrix3x3,
	pub translation: Matrix1x3
}

impl Extrinsic {
	pub fn new() -> Extrinsic {
		Extrinsic {
			rotation: Matrix3x3::identity(),
			translation: Matrix1x3::identity()
		}
	}
}


pub struct Camera {
	intrinsic: Intrinsic,
	pub extrinsic: Extrinsic,
	image_size: (i32, i32)
}

impl Camera {
	pub fn new() -> Camera {
		let k = Intrinsic::new();
		let rt = Extrinsic::new();
		Camera {
			intrinsic: k,
			extrinsic: rt,
			image_size: (0,0)
		}
	}

	pub fn new_from(intrinsic: Intrinsic, image_size: (i32, i32) ) -> Camera {
		Camera {
			intrinsic: intrinsic,
			extrinsic: Extrinsic::new(),
			image_size: image_size
		}
	}

	pub fn intrinsic(&self) -> Intrinsic {
		self.intrinsic
	}
}

