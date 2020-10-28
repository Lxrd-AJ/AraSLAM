#ifndef _VIZ_
#define _VIZ_

#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "./../ara_slam.h"

namespace ara_slam { namespace viz {
    void show_point_cloud(std::string, cv::Mat colors, cv::Mat depth);
    pcl::PointCloud<pcl::PointXYZ>::Ptr make_pointcloud(std::vector<cv::Point3f>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr make_pointcloud(std::vector<cv::Vec3f>);
    pcl::visualization::PCLVisualizer::Ptr init_pclviewer(std::string name);
}}


#endif