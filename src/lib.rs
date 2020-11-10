
use opencv::{
	prelude::*,
	core,
	highgui,
	videoio
};

pub mod data;
pub mod features;





pub fn add(a:i32, b:i32) -> i32 {
	a + b
}

pub fn test_module_system() {
	data::kitti::hello_world();
}

pub fn test_new_tsukuba_ds() {
	let _e = data::new_tsukuba::Lighting::Daylight;
}

pub fn show_window() -> opencv::Result<()> {
	let window = "Video Capture";
	highgui::named_window(window,1)?;
	let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
	let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    loop {
        let mut frame = core::Mat::default()?;
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
	}
	
	Ok(())
}


#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*; 
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
	}
	
	mod addition_tests {
		use super::*;

		#[test]
		fn test_add() {
			assert_eq!(add(4,5), 9);
		}
	}
}
