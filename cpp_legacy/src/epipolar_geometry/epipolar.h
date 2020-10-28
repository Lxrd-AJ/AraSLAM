/** \file epipolar.h
 *  \brief Epipolar Geometry Module
 * */ 

#ifndef _EPIPOLAR_H_
#define _EPIPOLAR_H_

#include <opencv2/opencv.hpp>
#include "./../ara_slam.h"

namespace ara_slam { namespace epipolar {
    /**
     * \brief Recover camera pose using the essential matrix
     * 
     * \details Compares the two sets of points and uses the intrinsic parameters of 
     * the camera to compute the camera pose. It does it by
     *  - compute the essential matrix E
     *  - compute the symmetric error of E as e
     *  - Decompose E into R and t 
     * 
     * \param[in]   a   keypoints deteced in image 1
     * \param[in]   b   keypoints deteced in image 2
     * 
     * \return  Tuple of results
     * \retval  cv::Mat The camera rotation
     * \retval  cv::Mat The camera translation
     * \retval  double  The symmetric reprojection error using the essential matrix
     * \retval  std::vector<int>    A array of inliers ints which are indices into `a` and `b` that determine which
     * points were inliers
     * 
     * */
    std::tuple<cv::Mat,cv::Mat,double,std::vector<int>> recover_pose_essential(std::vector<cv::Point2f>a,std::vector<cv::Point2f>b,cv::Mat);

    int recover_pose_homography(std::vector<cv::Point2f>,std::vector<cv::Point2f>,cv::Mat);

}}


#endif