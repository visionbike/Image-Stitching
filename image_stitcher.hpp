#pragma once
#ifndef __IMAGE_STITCHER_HPP__
#define __IMAGE_STITCHER_HPP__

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define MAX_RANSAC_ITERATIONS 5000
#define T_DIST 5  // threshold for distance in RANSAC algorithm

/**
 * Stitch two corresponding images together using RANSAC
 *
**/
class ImageStitcher
{
public:
    ImageStitcher() = default;
    ~ImageStitcher() = default;
    /**
     * Find the best homography matrix from two point matches of two images using RANSAC.
     * 1) Pick four random point pairs of two images.
     * 2) Estimate the homography H using just these four point pairs.
     * 3) Find inliers where distance(p', H*p) < threshold.
     * 4) Keep H if the inlier set is the largest.
     * 5) Repeat for another random four. Do this a maximum number of times, and remember the best.
     * The loop iteration N is determined adaptively.
     * N > log(1-p)/log(1-(1-e)^s)
     * where:
     * s - the number of points to compute solution
     * p - pobability of success
     * e = proportion outlier --> proportion of inlier is (1 - e)
     * Recompute least-squares H estimated on the inlier set
     * @param _pts1: corresponding points from the left image.
     * @param _pts2: corresponding points from the right image.
     * @param _maxIter: maximum iterations for RANSAC.
     * @return: the output estimated homography.
     */
    static cv::Mat findHomography(const std::vector<cv::Point2f> &_pts1,
                                  const std::vector<cv::Point2f> &_pts2,
                                  int _maxIter = MAX_RANSAC_ITERATIONS);

    /**
     * Stitch two images
     * @param _img1: the left image.
     * @param _img2: the right image.
     * @param _H: the estimated homography.
     * @param _dst: the output image.
     */
    static void stitch(const cv::Mat &_img1,
                       const cv::Mat &_img2,
                       const cv::Mat &_H,
                       cv::Mat &_dst);
private:
    /**
     * Get random four indices of matches to do homography.
     * @param _numMatches: number of the point matches.
     * @param _i1,_i2,_i3,_i4: four output indices.
     */
    static void _getRandomFourIndices(int _numMatches,
                                      int &_i1, int &_i2, int &_i3, int &_i4);

    /**
     * Get normalization matrix from points
     * @param _pts: corresponding points
     * @return: output normalization matrix
     */
    static cv::Mat _getNormMatrix(const std::vector<cv::Point2f> &_pts);

    /**
     * Estimate homography using SVD from given four corresponding point pairs.
     * To construct homography matrix A
     * [ -u1  -v1  -1   0    0    0   u1u2  v1u2  u2]
	 * [  0    0    0  -u1  -v1  -1   u1v2  v1v2  v2] * h = 0
	 * ................................................
	 * [  0    0    0  -u4  -v4  -1   u4v'4  v4v'4  v'4]
	 * where x' = Hx and h = [h1 ... h9] as a vector
     *
     * Use Singular Value Decomposition to compute A:
	 * UDV^T = A
     * h = V_smallest (column of V corresponding to smallest singular value).
	 * Then form H out of that.
     * @param _src: corresponding points from the source image.
     * @param _dst: corresponding points from the destination image.
     * @return: the output estimated homography matrix.
     */
    static cv::Mat _getHomographyFourPointPairs(const std::vector<cv::Point2f> &_src,
                                                const std::vector<cv::Point2f> &_dst);

    /**
     * Estimate homography using Least Square Optimization from inliers
     * @param _src: corresponding points from the source image.
     * @param _dst: corresponding points from the destination image.
     * @return: the output estimated homography matrix.
     */
    static cv::Mat _getHomographyLeastSquare(const std::vector<cv::Point2f> &_src,
                                             const std::vector<cv::Point2f> &_dst);

    /**
     * Compute inliers, given two set of points.
     * The homography transforms from the right image to the left one.
     * @param _src: corresponding points from the source image.
     * @param _dst: corresponding points from the destination image.
     * @param _inlierIndices: the list of inlier indices
     * @return:
     */
    static void _computeInliers(const std::vector<cv::Point2f> &_src,
                                const std::vector<cv::Point2f> &_dst,
                                const cv::Mat &_H,
                                std::vector<int> &_inlierIndices);

    /**
     * Interpolate image value at (x, y) position using Bilinear.
     * @param _img: reference image.
     * @param _x: x position.
     * @param _y: y position.
     * @return: interpolated value
     */
    static cv::Vec3b _interpolateBilinear(const cv::Mat &_img, const float &_x, const float &_y);
};

#endif //__IMAGE_STITCHER_H__
