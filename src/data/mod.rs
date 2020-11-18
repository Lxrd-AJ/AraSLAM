pub mod kitti;
pub mod new_tsukuba;

pub trait DataLoader : Iterator {
	fn length(&self) -> usize;
	fn download_url(&self) -> String;
	// fn read(&self, idx: usize) -> T;
}

pub trait StereoDataLoader<T> {
	fn read(&self, idx: usize) -> (T,T);
}

pub trait MonocularDataLoader<T> {
	fn read(&self, idx: usize) -> T;
}