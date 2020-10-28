
#include <opencv2/opencv.hpp>
// #include <opencv2/viz.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "./../ara_slam.h"

namespace ara_slam { namespace viz {
    //TODO: Add code to show a 3D point cloud
    void show_point_cloud(std::string, cv::Mat colors, cv::Mat depth){
        // cv::Mat point_cloud = cv::Mat::zeros(colors.rows, colors.cols, CV_32FC3);
        const int size = colors.rows * colors.cols;
        std::vector<cv::Vec3f> point_cloud(size);
        for(int idx = 0; idx < point_cloud.size(); idx++){
            for(int row = 0; row < colors.rows; row++){
                for(int col = 0; col < colors.cols; col++){
                    point_cloud[idx] = cv::Vec3f(row,col,depth.at<int>(row,col));
                    std::cout << point_cloud[idx] << std::endl;
                }
            }        
        }

        // pcl::PointCloud<pcl::PointXYZ> cloud;
        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        cloud->width = colors.cols;
        cloud->height = colors.rows;
        cloud->is_dense = true;
        cloud->points.resize(point_cloud.size());
        for(size_t i = 0; i < cloud->points.size(); i++){
            cloud->points[i].x = point_cloud[i][0];
            cloud->points[i].y = point_cloud[i][1];
            cloud->points[i].z = point_cloud[i][2];
        }
        pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
        // viewer.showCloud (cloud);
        // while (!viewer.wasStopped ())
        // {
        // }
    }

    /**
     * TODO:
     * - Add color to the point cloud      
     * */
    pcl::PointCloud<pcl::PointXYZ>::Ptr make_pointcloud(std::vector<cv::Point3f> points){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        cloud->width = points.size();
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(points.size());
        for(size_t i = 0; i < cloud->points.size(); i++){            
            cloud->points[i].x = points[i].x;
            cloud->points[i].y = points[i].y;
            cloud->points[i].z = points[i].z;
        }        
        return cloud;
        // viewer->addCoordinateSystem (1.0);
        // viewer->initCameraParameters ();
        
        // viewer.showCloud (cloud);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr make_pointcloud(std::vector<cv::Vec3f> points){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        cloud->width = points.size();
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(points.size());
        for(size_t i = 0; i < cloud->points.size(); i++){            
            cloud->points[i].x = points[i][1];
            cloud->points[i].y = points[i][0];
            cloud->points[i].z = points[i][2];
        }        
        return cloud;
    }

    pcl::visualization::PCLVisualizer::Ptr init_pclviewer(std::string name){
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (name));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, name);
        viewer->addCoordinateSystem(0.5);
        return viewer;
    }
}}

