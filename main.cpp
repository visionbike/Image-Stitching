#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "feature.hpp"
#include "image_stitcher.hpp"

void readPoints(const std::string &_fn, std::vector<cv::Point2f> &_pts)
{
    if (!_pts.empty()) { _pts.clear(); }
    std::fstream f;
    f.open(_fn, std::ios_base::in);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (line == "\n") { break; }
            std::istringstream in(line);
            float x = 0.f, y = 0.f;
            in >> x >> y;
            _pts.emplace_back(x, y);
        }
    }
    else {
        std::cout << "Cannot open the file!" << std::endl;
        exit(-1);
    }
    f.close();
}

void stitchTwoImages(const cv::Mat &_img1, const cv::Mat &_img2, const std::string &_ptfn1, const std::string &_ptfn2, const std::string &_fn)
{
    cv::Mat gray1, gray2;
    cv::cvtColor(_img1, gray1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(_img2, gray2, cv::COLOR_RGB2GRAY);
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    std::vector<cv::DMatch> matches, gmatches;

    detectSIFTFeature(gray1, kpts1, desc1);
    detectSIFTFeature(gray2, kpts2, desc2);

    matchFeature(desc1, desc2, matches);
    keepGoodMatches(matches, gmatches);

    std::vector<cv::Point2f> pts1, pts2;
    for (auto & match : gmatches) {
        cv::Point2f pt = kpts1[match.queryIdx].pt;
        pt.x = std::round(pt.x);
        pt.y = std::round(pt.y);
        pts1.push_back(pt);
        pt = kpts2[match.trainIdx].pt;
        pt.x = std::round(pt.x);
        pt.y = std::round(pt.y);
        pts2.push_back(pt);
    }

    // Draw corresponding points
    cv::Mat drawImg1, drawImg2;
    _img1.copyTo(drawImg1);
    _img2.copyTo(drawImg2);
    for (int i = 0; i < pts1.size(); ++i) {
        cv::circle(drawImg1, pts1[i], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(drawImg2, pts2[i], 2, cv::Scalar(0, 255, 0), 2);
    }

    // Save drawn images
    cv::imwrite("v" + _ptfn1 + ".JPG", drawImg1);
    cv::imwrite("v" + _ptfn2 +".JPG", drawImg2);

    // Write to file
    std::fstream f1, f2;
    f1.open(_ptfn1 + ".txt", std::ios_base::out);
    f2.open(_ptfn2 + ".txt", std::ios_base::out);

    if (f1.is_open() && f2.is_open()) {
        for (int i = 0; i < pts1.size(); ++i) {
            f1 << (int)pts1[i].x << " " << (int)pts1[i].y << std::endl;
            f2 << (int)pts2[i].x << " " << (int)pts2[i].y << std::endl;
        }
    }

    f1.close();
    f2.close();

    cv::Mat result;
    cv::Mat H = ImageStitcher::findHomography(pts1, pts2);
    ImageStitcher::stitch(_img1, _img2, H, result);
    cv::imwrite(_fn + ".JPG", result);
}

void stitchTwoImages2(const cv::Mat &_img1, const cv::Mat &_img2, const std::string &_ptfn1, const std::string &_ptfn2, const std::string &_fn)
{
    std::vector<cv::Point2f> _pts1, _pts2;
    readPoints(_ptfn1 + ".txt", _pts1);
    readPoints(_ptfn2 + ".txt", _pts2);

    // Draw corresponding points
    cv::Mat drawImg1, drawImg2;
    _img1.copyTo(drawImg1);
    _img2.copyTo(drawImg2);
    for (int i = 0; i < _pts1.size(); ++i) {
        cv::circle(drawImg1, _pts1[i], 2, cv::Scalar(0, 255, 0), 2);
        cv::circle(drawImg2, _pts2[i], 2, cv::Scalar(0, 255, 0), 2);
    }

    // Save drawn images
    cv::imwrite("v" + _ptfn1 + ".JPG", drawImg1);
    cv::imwrite("v" + _ptfn2 + ".JPG", drawImg2);

    cv::Mat result;
    cv::Mat H = ImageStitcher::findHomography(_pts1, _pts2);
    ImageStitcher::stitch(_img1, _img2, H, result);
    cv::imwrite(_fn + ".JPG", result);
}

int main() {
    std::string imgRoot = "images";
    int numImages = 4;

    // Load images
    // Convert to grayscale
    std::vector<cv::Mat> rgbImages;
    for (int i = numImages; i >= 1; --i) {
        cv::Mat rgb = cv::imread( imgRoot + "/00" + std::to_string(i) + ".JPG", cv::IMREAD_COLOR);
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        rgbImages.push_back(rgb);
    }

    stitchTwoImages(rgbImages[0], rgbImages[1], "0004", "0003", "0043");
    cv::Mat img43 = cv::imread("0043.JPG", cv::IMREAD_COLOR);
    stitchTwoImages(img43, rgbImages[2], "0043", "0002", "0432");
    cv::Mat img432 = cv::imread("0432.JPG", cv::IMREAD_COLOR);
    stitchTwoImages(img432, rgbImages[3], "0432", "0001", "4321");
    return 0;
}
