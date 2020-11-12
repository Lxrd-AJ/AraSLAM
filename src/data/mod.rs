pub mod kitti;
pub mod new_tsukuba;

pub trait DataLoader {
	fn length(&self) -> usize;
	fn download_url(&self) -> String;
}