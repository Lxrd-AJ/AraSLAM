#include <opencv2/opencv.hpp>
#include "./../ara_slam.h"
#include <cassert>
#include <limits> 

namespace ara_slam { namespace epipolar {

    //---------------- PRIVATE FUNCTIONS ----------------------


    /**
     * This error is calculated by exploiting the epipolar constraint of point
     * correspondence `xFx' = 0`; i.e the points `x` and `x'` in the two images must 
     * lie on the same epipolar line.
     * Therefore to calculate the symmetric error, we must compute the epiopolar error for 
     * all pairs of points in our inliers.
     * 
     * @return a tuple of double e and vector b
     *      - e is the total symmetric error computed from the epipolar error of the inliers
     *      - b is a vector of integers containing the filtered inliers 
     */
    std::tuple<double,std::vector<int>> symmetric_error_essential(cv::Mat E, cv::Mat K, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, std::vector<int> inliers_idx){
        //derive the fundamental matrix from the essential matrix        
        cv::Mat F = K.inv().t() * E * K.inv();        
        double sigma = 1.0;
        const double inv_sigma_sq = 1.0 / pow(sigma,2);
        const double th = 3.841;
        const double th_score = 5.991;
        double error = 0.0;
        std::vector<int> better_inliers;
        //loop through the point pairs and calculate their reprojection error using the 
        //fundamental matrix
        assert( inliers_idx.size() > 0 );
        for(size_t i = 0; i < inliers_idx.size(); ++i){
            const int idx = inliers_idx[i];
            const cv::Point2f p1 = points1[idx];
            const cv::Point2f p2 = points2[idx];
            bool accept_point = true;
            cv::Mat x = cv::Mat::ones(3,1,CV_64F);
            cv::Mat x_h = cv::Mat::ones(3,1,CV_64F);
            x.at<double>(0,0) = p1.x; x.at<double>(0,1) = p1.y; 
            x_h.at<double>(0,0) = p2.x; x_h.at<double>(0,1) = p2.y; 
            
            //calculate the reprojection error of `x` in image 2 using the epipolar constraints
            // epipolar_line_img_2 * x = Fx
            // x_hat * F * x = 0
            const cv::Mat l2 = F * x;                        
            const cv::Mat e2 = x_h.t() * l2;
            //using the method in orb_slam; see https://github.com/felixchenfy/Monocular-Visual-Odometry/ and orb slam https://github.com/raulmur/ORB_SLAM
            const double l2_sq = pow(l2.at<double>(0,0),2) + pow(l2.at<double>(0,1),2);
            const cv::Mat sq_dist_1 = e2 * e2 / l2_sq;            
            const cv::Mat _chi_sq = sq_dist_1 * inv_sigma_sq;
            const double chi_sq = _chi_sq.at<double>(0,0);            

            //calculate the reprojection error of `x_hat` in image 1 using the epipolar constraints
            const cv::Mat l1 = F * x_h;
            const cv::Mat e1 = x.t() * l1;
            // const double l1_sq = l1.dot(l1);((cv::Mat)(l1.t() * l1)).at<double>(0,0);
            const double l1_sq = pow(l1.at<double>(0,0),2) + pow(l1.at<double>(0,1),2);
            const double sq_dist_2 = ((cv::Mat)(e1 * e1 / l1_sq)).at<double>(0,0);
            const double chi_sq_2 = sq_dist_2 * inv_sigma_sq;            

            if( (chi_sq <= th) || (chi_sq_2 <= th) ){
                if(chi_sq <= th) error += th_score - chi_sq;
                if(chi_sq_2 <= th) error += th_score - chi_sq_2;
                if(chi_sq <= th && chi_sq_2 <= th) better_inliers.push_back(i);
            }
        }//end for loop        
        return std::make_tuple(error, better_inliers);
    }

    cv::Point2f camera_norm_plane(cv::Point2f p, cv::Mat K){
        const auto x =  (p.x - K.at<double>(0,2)) / K.at<double>(0,0);
        const double y = (p.y - K.at<double>(1,2)) / K.at<double>(1,1);
        return cv::Point2f(x,y);
    }




    //---------------- PUBLIC FUNCTIONS ----------------------
    /**
     * Estimates the rotation and translation of the camera based on the homography
     * between the planes from which points `p1` and `p2` were derived from
     */
    int recover_pose_homography(std::vector<cv::Point2f> p1,std::vector<cv::Point2f> p2, cv::Mat K){
        //estimates the homography matrix
        cv::Mat inliers_mask;
        std::vector<int> inliers;
        cv::Mat H = cv::findHomography(p1,p2,cv::RANSAC,3,inliers_mask);
        //Populate `inliers` using the `inliers_mask`
        for(int idx=0; idx < inliers_mask.rows; ++idx){
            bool inlier = (int)inliers_mask.at<unsigned char>(idx,0) == 1 ? true : false;
            if(inlier){ inliers.push_back(idx); }
        }
        //Recover the set of rotations and translations
        std::vector<cv::Mat> rotations, translations, normals;
        cv::decomposeHomographyMat(H,K,rotations,translations,normals);
        //transform `p1` and `p2` into points on the normalised plane so that they can be 
        //used to filter the 2 wrong solutions out of the 4 returned from `decomposeHomographyMat`
        std::vector<cv::Point2f> norm_p1, norm_p2;
        std::transform(p1.begin(), p1.end(), std::back_inserter(norm_p1), 
            [K](cv::Point2f p) -> cv::Point2f {
                return camera_norm_plane(p,K);
            });
        std::transform(p2.begin(), p2.end(), std::back_inserter(norm_p2), 
            [K](cv::Point2f p) -> cv::Point2f {
                return camera_norm_plane(p,K);
            });
        //using the points on the normalised plane, filter them out so that we know which
        //normalised points are in our inliers mask
        std::vector<cv::Point2f> inliers_norm_p1, inliers_norm_p2;
        for(int idx: inliers){
            inliers_norm_p1.push_back( norm_p1[idx] );
            inliers_norm_p2.push_back( norm_p2[idx] );
        }
        //filter the set of solutions
        std::vector<int> solutions;
        std::vector<cv::Mat> p_rotations, p_translations, p_normals;
        cv::filterHomographyDecompByVisibleRefpoints(
            rotations,normals,inliers_norm_p1,inliers_norm_p2,solutions );
        std::cout << "Found " << solutions.size() << " solutions " << std::endl;
        for(int idx: solutions){
            p_rotations.push_back(rotations[idx]);
            p_translations.push_back(translations[idx]);
            p_normals.push_back(normals[idx]);
        }
        std::cout << p_rotations << std::endl;
        std::cout << p_translations << std::endl;
        std::cout << p_normals << std::endl;
        exit(0);
    }



    std::tuple<cv::Mat,cv::Mat,double,std::vector<int>> 
        recover_pose_essential(
            std::vector<cv::Point2f> p1,std::vector<cv::Point2f> p2, cv::Mat K){ 
        cv::Mat R,t, mask;
        std::vector<int> inliers_idx;
        const cv::Mat E = cv::findEssentialMat(p1,p2,K,cv::RANSAC,0.999,1.0,mask);

        //TODO: Figure out if I need to normalise the essential matrix
        //NB: So far based on my experiments, there has been no need to normalise E
        // E /= E.at<double>(2,2);
        if(mask.empty() || E.empty() || E.rows > 3){
            return std::make_tuple(R,t,std::numeric_limits<double>::max(),inliers_idx);
        }
        cv::recoverPose(E,p1,p2,K,R,t,mask);
        
        //get the inliers using `mask`
        for(int idx=0; idx < mask.rows; ++idx){
            bool inlier = (int)mask.at<unsigned char>(idx,0) == 1 ? true : false;
            if(inlier){ inliers_idx.push_back(idx); }
        }
        
        //TODO: Determine if I need to normalise the translation
        //       t = t / sqrt(t.at<double>(1, 0) * t.at<double>(1, 0) + t.at<double>(2, 0) * t.at<double>(2, 0) + t.at<double>(0, 0) * t.at<double>(0, 0));
        double error; std::vector<int> new_inliers;
        std::tie(error, new_inliers) = symmetric_error_essential(E,K,p1,p2,inliers_idx);
        return std::make_tuple(R,t,error,new_inliers);
    }



}}


//TODO: `triangulation` etc