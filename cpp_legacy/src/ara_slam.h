/**
 * @defgroup group1 The First Group
 */

#ifndef _ARA_SLAM_H
#define _ARA_SLAM_H

#include <string>
#include <opencv2/opencv.hpp>


/**
 * Global classes and functions go here 
 **/
namespace ara_slam {
    const std::string version("0.0.1");

    

    enum class feature_extractor {
        sift,
        orb,
        fast
    };

    //TODO: Add a macro here to support optional flags for compilation
    // C++ template to print vector container elements 
    template <typename T> 
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) { 
        os << "["; 
        for (int i = 0; i < v.size(); ++i) { 
            os << v[i]; 
            if (i != v.size() - 1) 
                os << ", "; 
        } 
        os << "]";   
        return os; 
    }
    enum class feature_matcher {
        brute,
        flann,
        knn
    };

    enum class odometry_state {
        limbo,
        initialising,
        tracking,
        lost
    };

    //TODO
    //Write a resize function similar to cv::resize but accept a camera matrix as an optional parameter
    // inline resize(cur_frame, cur_frame, cv::Size(), 0.50, 0.50, intrinsic_matrix<optional>);

    inline cv::Mat hconcat(cv::Mat img1, cv::Mat img2){
        cv::Mat result;
        if( img1.channels() <= 2){
            cv::hconcat( img1, img2, result);
        }else{ //assume 3 channels            
            std::vector<cv::Mat> channels1(3);
            std::vector<cv::Mat> channels2(3);
            cv::split(img1, channels1);
            cv::split(img2, channels2);

            cv::Mat B, G, R;
            cv::hconcat( channels1[0], channels2[0], B); //B
            cv::hconcat( channels1[1], channels2[1], G); //G
            cv::hconcat( channels1[2], channels2[2], R); //R
            
            std::vector<cv::Mat> x { B, G, R };
            cv::merge( x, result);
        }
        return result;
    }

    inline cv::Mat vconcat(cv::Mat img1, cv::Mat img2){
        cv::Mat result;
        if( img1.channels() <= 2){
            cv::vconcat( img1, img2, result);
        }else{ //assume 3 channels            
            std::vector<cv::Mat> channels1(3);
            std::vector<cv::Mat> channels2(3);
            cv::split(img1, channels1);
            cv::split(img2, channels2);

            cv::Mat B, G, R;
            cv::vconcat( channels1[0], channels2[0], B); //B
            cv::vconcat( channels1[1], channels2[1], G); //G
            cv::vconcat( channels1[2], channels2[2], R); //R
            
            std::vector<cv::Mat> x { B, G, R };
            cv::merge( x, result);
        }
        return result;
    }

    inline cv::Mat make_anaglyph(cv::Mat first, cv::Mat second){
        cv::Mat result;
        if( first.channels() < 3){
            std::cerr << "Unimplemented for grayscale images " << std::endl;
            exit(0);
        }else{
            std::vector<cv::Mat> first_channels(3);
            std::vector<cv::Mat> sec_channels(3);
            cv::split(first, first_channels);
            cv::split(second, sec_channels);
            
            std::vector<cv::Mat> x{ sec_channels[0],sec_channels[1],first_channels[2] }; //B,G,R
            cv::merge(x,result);
        }
        return result;
    }
}

#endif


/**
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+

**/