
#include <fstream>
#include <iostream>
#include <dirent.h>
#include "utils.h"

cv::Mat create_restricted_area_on_src_img(const cv::Mat &src){

    const int r = 100;
    const unsigned int POLYGON_POINT_NUM = 6; 
    const double shift_x = 100;
    const double shift_y = 100;

    cv::Mat mask = cv::Mat::zeros( cv::Size( src.rows, src.cols), CV_8U );
    std::vector<cv::Point2f> vert(POLYGON_POINT_NUM);
    vert[0] = cv::Point(3*r/2 + shift_x, static_cast<int>(1.34*r + shift_y) );
    vert[1] = cv::Point( 1*r + shift_x, 2*r + shift_y);
    vert[2] = cv::Point( 3*r/2 + shift_x, static_cast<int>(2.866*r + shift_y) );
    vert[3] = cv::Point( 5*r/2 + shift_x, static_cast<int>(2.866*r + shift_y) );
    vert[4] = cv::Point( 3*r + shift_x, 2*r + shift_y);
    vert[5] = cv::Point( 5*r/2 + shift_x, static_cast<int>(1.34*r + shift_y) );
    for( int i = 0; i < POLYGON_POINT_NUM; i++ )
    {
        cv::line( mask, vert[i] ,  vert[(i+1)%6] , cv::Scalar( 255 ), 3 );
    }
    
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat raw_dist( src.size(), CV_8U );
    for( int i = 0; i < src.rows; i++ )
    {
        for( int j = 0; j < src.cols; j++ )
        {
            float check = (float)cv::pointPolygonTest( contours[0], cv::Point2f((float)j, (float)i), true );
            if ( check >=0){
                raw_dist.at<unsigned char>(i,j) = (unsigned char)255;
            }
            else{
                raw_dist.at<unsigned char>(i,j) = (unsigned char)0;
            }
            
        }
    }
    return raw_dist;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

bool inside_restricted_area(const std::vector<std::vector<cv::Point> > &contours, const cv::Rect &box){
    for( int i = box.x; i < box.x + box.width; i++ )
    {
        for( int j = box.y; j < box.y + box.height; j++ )
        {
            if( (float)cv::pointPolygonTest( contours[0], cv::Point2f((float)j, (float)i), true ) > 0){
                return true;
            }
        }
    }
    return false;
}