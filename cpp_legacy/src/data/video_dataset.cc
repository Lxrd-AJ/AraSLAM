#include <opencv2/opencv.hpp>
// #include <opencv2/sfm/projection.hpp>
// #include <boost/range/iterator_range.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/algorithm/string.hpp>
// #include <tuple>
// #include <vector>
// #include <fstream>
#include "./../ara_slam.h"
#include "dataset.h"
#include "video_dataset.h"


using namespace boost::filesystem;

namespace ara_slam{ namespace data {

	VideoDataset::VideoDataset(std::string ref){
		filename = ref;
		capture = cv::VideoCapture(ref);
		fps = capture.get(cv::CAP_PROP_FPS);
		frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
	}

	VideoDataset::~VideoDataset(){
		capture.release();
	}

	int VideoDataset::length(){
		return frame_count;
	}

	float VideoDataset::duration(){
		return (float)frame_count / (float)fps;
	}

	cv::Mat VideoDataset::stream(){
		cv::Mat frame;
		capture >> frame;
		return frame;
	}

	cv::Mat VideoDataset::operator[](int index){
		cv::Mat frame;
		capture.set(cv::CAP_PROP_POS_FRAMES, index);
		capture >> frame;
		return frame;
	}
}}