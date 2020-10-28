#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <Eigen/Dense>
#include <numeric>
#include <cassert>
#include <set>
#include <iostream>
#include "./../ara_slam.h"

namespace ara_slam { namespace localisation {
    using features_t = std::tuple<std::vector<cv::KeyPoint>,cv::Mat>;

    //TODO: Remove this later as it has been deprecated
    std::tuple<cv::Mat,cv::Mat> recover_pose(std::vector<cv::Point2f> p1,std::vector<cv::Point2f> p2,cv::Mat K){        
        cv::Mat R,t, mask;
        cv::Mat E = cv::findEssentialMat(p1,p2,K,cv::RANSAC,0.999,1.0,mask);
        cv::recoverPose(E,p1,p2,K,R,t,mask);
        return std::make_tuple(R,t);
    }


    std::tuple<std::vector<cv::Point2f>,std::vector<cv::Point2f>> track_keypoints(cv::Mat first,std::vector<cv::Point2f> points,cv::Mat second){
        if( points.size() == 0 ){
            std::cout << "Cannot track an empty vector of points" << std::endl;
        }
        std::vector<uchar> status;
        std::vector<float> error;
        std::vector<cv::Point2f> f_first, f_second;
        std::vector<cv::Point2f> predicted;
        // cv::KeyPoint::convert(points, kps);
        cv::calcOpticalFlowPyrLK(first, second, points, predicted, status, error);
        
        for(size_t i=0; i < status.size(); i++){
            cv::Point2f pred_pt = predicted[i];
            if( status[i] == 1 && (pred_pt.x >= 0 && pred_pt.y >= 0) ){
                f_first.push_back( points[i] );
                f_second.push_back( predicted[i] );
            }
        }
        
        return std::make_tuple(f_first, f_second);
    }

    std::vector<cv::KeyPoint> filter_keypoints(std::vector<cv::KeyPoint> kps, std::vector<bool> matches){
        std::vector<cv::KeyPoint> result;
        for(auto &kp: kps){
            if( matches[&kp - &kps[0]] ){
                result.push_back(kp);
            }
        }
        return result;
    }

    std::vector<cv::Point3i> xyd_from_keypoints(std::vector<cv::KeyPoint> kps, cv::Mat depth_map){
        std::vector<cv::Point3i> point_cloud;
        for(const auto kp: kps){
            cv::Point x_y = kp.pt;
            cv::Point3i p = cv::Point3i(x_y);
            p.z = depth_map.at<uchar>(x_y.y, x_y.x);
            point_cloud.push_back(p);            
        }
        return point_cloud;
    }

    std::vector<int> max_clique(std::vector<cv::Point3f> t_1, std::vector<cv::Point3f> t){
        assert(t_1.size() == t.size() && "Both keypoints must be equal");
        Eigen::MatrixXd M_1(t.size(),t.size());        
        Eigen::MatrixXd M(t.size(), t.size());
        Eigen::MatrixXi cM = Eigen::MatrixXi::Zero(t.size(),t.size());
        for(size_t i = 0; i < t.size(); ++i){
            for(size_t j=0; j < t.size(); ++j){
                M_1(i,j) = cv::norm(t_1[i] - t_1[j]);
                M(i,j) = cv::norm(t[i] - t[j]);
            }
        }
        Eigen::MatrixXd Q = M_1 - M;
        Q = Q.cwiseAbs();        
        for(Eigen::Index idx=0; idx < Q.size(); ++idx){
            if( Q(idx) < 0.5 ){                
                cM(idx) = 1;
            }
        }
        
        Eigen::VectorXi node_lengths = cM.rowwise().sum();
        Eigen::VectorXi::Index max_index;        
        int max_count = node_lengths.maxCoeff(&max_index);              
        std::set<int> clique = { (int)max_index };

        do{
            std::set<int> potential_nodes;            
            for(int row=0; row < t.size(); ++row){ //for all nodes
                //check if this current node `i` is connected to all the nodes in our clique
                bool connected = std::accumulate(clique.begin(), clique.end(), true, [cM,row](bool e, int cq){
                    return e && cM(row,cq);
                });
                //if connected and is not in the clique; then it is a potential node
                if(connected && (clique.find(row) == clique.end()) ){ potential_nodes.insert(row); }                
            }
            
            //Find the node in potential_nodes that has the most connection to other nodes in potential nodes            
            int max_node = std::accumulate(potential_nodes.begin(), potential_nodes.end(), 0, 
                [max_node,cM](int acc_node, int node){
                    int count = cM.row(node).sum();                    
                    if( count > acc_node ){ return node; }                    
                    return acc_node;
            }); 
            int max_count = cM.row(max_node).sum();

            if( max_count == 0 || clique.size() >= t.size() || potential_nodes.size() == 0 ){                
                break;
            }else{
                clique.insert(max_node);
            }               
        }while(true);        

        // std::vector<cv::Point3f> results;
        // std::transform(clique.begin(), clique.end(), std::back_inserter(results), [t](int idx) -> cv::Point3f {
        //     return t[idx];
        // }); 
        return std::vector<int>(clique.begin(), clique.end());
    }
}}
