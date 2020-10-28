#ifndef _DATASET_H_
#define _DATASET_H_

#include "./../ara_slam.h"

namespace ara_slam{ namespace data {
    /**
     * Inspired by the PyTorch dataset class https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
     * */
    template <typename T>
    class Dataset {        
        virtual int length() = 0;
        virtual T operator[](int) = 0;
        };
}}

#endif