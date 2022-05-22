#ifndef __TRT_UTILS_H_
#define __TRT_UTILS_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

// define the polygon shape of the restriction area, based on the size of the source image
cv::Mat create_restricted_area_on_src_img(const cv::Mat &src);
// read all image files in the working directory
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
// verify whether the detected object stays inside the restriction region based on the contours 
bool inside_restricted_area(const std::vector<std::vector<cv::Point> > &contours, const cv::Rect &box);



#endif







