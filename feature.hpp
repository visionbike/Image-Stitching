#pragma once
#ifndef __FEATURE_H__
#define __FEATURE_H__

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

void detectSIFTFeature(const cv::Mat &_img, std::vector<cv::KeyPoint> &_kpts, cv::Mat &_desc);
void matchFeature(const cv::Mat &_desc1, const cv::Mat &_desc2, std::vector<cv::DMatch> &_matches);
void keepGoodMatches(const std::vector<cv::DMatch> &_matches, std::vector<cv::DMatch> &_gmatches);

#endif //__FEATURE_H_
