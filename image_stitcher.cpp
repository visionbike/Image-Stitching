#include "image_stitcher.hpp"

cv::Mat ImageStitcher::findHomography(const std::vector<cv::Point2f> &_pts1,
                                      const std::vector<cv::Point2f> &_pts2,
                                      int _maxIter)
{
    CV_Assert(_pts1.size() == _pts2.size());
    // Initialize random state
    srand(time(nullptr));

    // Number of corresponding points
    int numPts = _pts1.size();
    // Four random indices
    int i1= 0, i2 = 0, i3 = 0, i4 = 0;
    // Four corresponding point pairs to do homography
    std::vector<cv::Point2f> src(4), dst(4);
    // Inlier indices
    std::vector<int> inlierIndices;
    // Max number of inliers
    int maxNumInliers = 0;
    cv::Mat bestH;

    // RANSAC algorithm
    for (int i = 0; i < _maxIter; ++i) {
        // Choose four random corresponding points
        _getRandomFourIndices(numPts, i1, i2, i3, i4);
        src[0] = _pts2[i1];
        src[1] = _pts2[i2];
        src[2] = _pts2[i3];
        src[3] = _pts2[i4];

        dst[0] = _pts1[i1];
        dst[1] = _pts1[i2];
        dst[2] = _pts1[i3];
        dst[3] = _pts1[i4];

        cv::Mat H = _getHomographyFourPointPairs(src, dst);
        if (H.empty()) { continue; }

        // Compute inliers
        _computeInliers(_pts2, _pts1, H, inlierIndices);
        if (inlierIndices.size() > maxNumInliers) {
            maxNumInliers = inlierIndices.size();
            bestH = H;
        }
    }

    std::cout << "[INFO] Maximum inliers: "<< maxNumInliers << std::endl;

    std::vector<cv::Point2f> srcInliers, dstInliers;
    for (auto & i : inlierIndices) {
        srcInliers.push_back(_pts2[i]);
        dstInliers.push_back(_pts1[i]);
    }

    // Compute least square homography from inliers
    bestH = _getHomographyLeastSquare(srcInliers, dstInliers);
    return bestH;
}

cv::Vec3b ImageStitcher::_interpolateBilinear(const cv::Mat &_img, const float &_x, const float &_y)
{
    float x1 = std::floor(_x);
    float x2 = std::ceil(_x);
    float y1 = std::floor(_y);
    float y2 = std::ceil(_y);

    cv::Vec3b y1Val = ((x2 - _x) / (x2 - x1)) * _img.at<cv::Vec3b>(y1, x1) + ((_x - x1) / (x2 - x1)) * _img.at<cv::Vec3b>(y1, x2);
    cv::Vec3b y2Val = ((x2 - _x) / (x2 - x1)) * _img.at<cv::Vec3b>(y2, x1) + ((_x - x1) / (x2 - x1)) * _img.at<cv::Vec3b>(y2, x2);

    cv::Vec3b val = ((y2 - _y) / (y2 - y1)) * y1Val + ((_y - y1) / (y2 - y1)) * y2Val;
    return val;
}

void ImageStitcher::stitch(const cv::Mat &_img1,
                           const cv::Mat &_img2,
                           const cv::Mat &_H,
                           cv::Mat &_dst)
{
    // Get top left, top right, bottom right, bottom left of _img2
    cv::Mat tl2 = (cv::Mat_<float>(3,1) << 0.f, 0.f, 1.f);
    cv::Mat tr2 = (cv::Mat_<float>(3,1) << _img2.cols, 0.f, 1.f);
    cv::Mat br2 = (cv::Mat_<float>(3,1) << _img2.cols, _img2.rows, 1.f);
    cv::Mat bl2 = (cv::Mat_<float>(3,1) << 0.f, _img2.rows, 1.f);

    tl2 = _H * tl2;
    tr2 = _H * tr2;
    br2 = _H * br2;
    bl2 = _H * bl2;

    // Get offset of _img1 from warped _img2
    int offsetY = (int)abs(std::min(0.f, std::min(tr2.at<float>(1), tl2.at<float>(1))));
    int offsetX = (int)abs(std::min(0.f, std::min(tl2.at<float>(0), bl2.at<float>(0))));

    //    Get final image size, by predicting the overlap and
    //    getting the largest rectangle that fits over this
    int height = (int)abs(std::max((float)_img1.rows, std::max(bl2.at<float>(1), br2.at<float>(1))) - std::min(0.f, std::min(tr2.at<float>(1), tl2.at<float>(1))));
    int width = (int)abs(std::max((float)_img1.cols, std::max(tr2.at<float>(0), br2.at<float>(0))) - std::min(0.f, std::min(tl2.at<float>(0), bl2.at<float>(0))));

    if (!_dst.empty()) { _dst.release(); }
    _dst = cv::Mat::zeros(height, width, CV_8UC3);

    // Add the first image to the centre of the new big Mat
    for (int y = 0; y < _img1.rows; ++y) {
        for (int x = 0; x < _img1.cols; ++x) {
                _dst.at<cv::Vec3b>(y + offsetY, x + offsetX) = _img1.at<cv::Vec3b>(y,x);
        }
    }

    // Warp the image2 and add to _dst also apply alpha blending
    float alpha = 0.2f;
    for (int y = 0; y < _dst.rows; ++y) {
        for (int x = 0; x < _dst.cols; ++x) {
            // Inverse mapping values from _dst to _img2
            cv::Mat p = (cv::Mat_<float>(3, 1) << x - offsetX, y - offsetY, 1);
            cv::Mat px = _H.inv() * p;
            px /= px.at<float>(2, 0);
            if ((0 < px.at<float>(0)) && (px.at<float>(0) < _img2.cols - 1)) {
                if ((0 < px.at<float>(1)) && (px.at<float>(1) < _img2.rows - 1)) {
                    cv::Vec3b pixelVal = _interpolateBilinear(_img2, px.at<float>(0), px.at<float>(1));
                    if (_dst.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) { _dst.at<cv::Vec3b>(y, x) = pixelVal; }
                    else { _dst.at<cv::Vec3b>(y, x) = (alpha * pixelVal + (1 - alpha) * _dst.at<cv::Vec3b>(y, x)); } // alpha blending
                }
            }
        }
    }

    // Crop the result image
    int st = -1, end = _dst.rows;
    for (int j = 0; j < _dst.cols; ++j) {
        int mnpos = _dst.rows, mxpos = -1;
        for (int i = 0; i < _dst.rows; ++i) {
            if(_dst.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) { continue; }

            mnpos = std::min(mnpos, i);
            mxpos = std::max(mxpos, i);
        }
        st = cv::max(st, mnpos);
        end = cv::min(end, mxpos);
    }
    cv::Mat crop = cv::Mat(end - st + 1, _dst.cols, CV_8UC3);
    for (int j = 0; j < _dst.cols; ++j) {
        for (int i = st; i <= end; ++i) {
            crop.at<cv::Vec3b>(i - st, j) = _dst.at<cv::Vec3b>(i, j);
        }
    }
    _dst = crop;
}

void ImageStitcher::_getRandomFourIndices(int _numMatches,
                                          int &_i1, int &_i2, int &_i3, int &_i4)
{
    _i1 = rand() % _numMatches;
    do { _i2 = rand() % _numMatches; } while (_i2 == _i1);
    do { _i3 = rand() % _numMatches; } while (_i3 == _i1 || _i3 == _i2);
    do { _i4 = rand() % _numMatches; } while (_i4 == _i1 || _i4 == _i2 || _i4 == _i3);
}

cv::Mat ImageStitcher::_getNormMatrix(const std::vector<cv::Point2f> &_pts)
{
    // Compute mean and std of _pts
    cv::Point2f mean(0.f, 0.f);
    for (auto & pt : _pts) { mean += pt; }
    mean /= (float)_pts.size();

    cv::Point2f std(0.f, 0.f);
    for (auto & pt : _pts) {
        cv::Point2f square = pt - mean;
        std += cv::Point2f(square.x * square.x, square.y * square.y);
    }
    std.x = std::sqrt(std.x);
    std.y = std::sqrt(std.y);
    std /= (float)_pts.size();

    cv::Mat normMat = (cv::Mat_<float>(3, 3) <<
            1.f / std.x, 0.f        , -1.f * mean.x / std.x,
            0.f        , 1.f / std.y, -1.f * mean.y / std.y,
            0.f        , 0.f        , 1.f);

    return normMat;
}

cv::Mat ImageStitcher::_getHomographyFourPointPairs(const std::vector<cv::Point2f> &_src,
                                                    const std::vector<cv::Point2f> &_dst)
{
    CV_Assert(_src.size() == _dst.size() && _src.size() == 4);
    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);

    // Construct A with size of 8x9
    cv::Mat A = cv::Mat::zeros(8, 9, CV_32FC1);

    for (int i = 0; i < 4; ++i) {
        float u1 = _src[i].x;
        float v1 = _src[i].y;

        float u2 = _dst[i].x;
        float v2 = _dst[i].y;

        A.at<float>(2 * i, 0) = -u1;
        A.at<float>(2 * i, 1) = -v1;
        A.at<float>(2 * i, 2) = -1.f;
        A.at<float>(2 * i, 6) = u1 * u2;
        A.at<float>(2 * i, 7) = v1 * u2;
        A.at<float>(2 * i, 8) = u2;
        
        A.at<float>(2 * i + 1, 3) = -u1;
        A.at<float>(2 * i + 1, 4) = -v1;
        A.at<float>(2 * i + 1, 5) = -1.f;
        A.at<float>(2 * i + 1, 6) = u1 * v2;
        A.at<float>(2 * i + 1, 7) = v1 * v2;
        A.at<float>(2 * i + 1, 8) = v2;
    }

    // Compute SVD
    cv::Mat S, U, Vt;
    cv::SVD::compute(A, S, U, Vt, cv::SVD::FULL_UV);

    // The H values are the last row of Vt
    for (int i = 0; i < 9; ++i) { H.at<float>(i / 3, i % 3) = Vt.at<float>(8, i); }

    // Normalize H
    H /= H.at<float>(2, 2);
    return H;
}

cv::Mat ImageStitcher::_getHomographyLeastSquare(const std::vector<cv::Point2f> &_src,
                                                 const std::vector<cv::Point2f> &_dst)
{
    cv::Mat X = cv::Mat::zeros(2 * _src.size(), 9, CV_32F);
    for (int i = 0; i < _src.size(); ++i) {
        float u1 = _src[i].x;
        float v1 = _src[i].y;

        float u2 = _dst[i].x;
        float v2 = _dst[i].y;

        X.at<float>(2 * i, 0) = -u1;
        X.at<float>(2 * i, 1) = -v1;
        X.at<float>(2 * i, 2) = -1.f;
        X.at<float>(2 * i, 6) = u1 * u2;
        X.at<float>(2 * i, 7) = v1 * u2;
        X.at<float>(2 * i, 8) = u2;

        X.at<float>(2 * i + 1, 3) = -u1;
        X.at<float>(2 * i + 1, 4) = -v1;
        X.at<float>(2 * i + 1, 5) = -1.f;
        X.at<float>(2 * i + 1, 6) = u1 * v2;
        X.at<float>(2 * i + 1, 7) = v1 * v2;
        X.at<float>(2 * i + 1, 8) = v2;
    }

    cv::Mat S, U, Vt;
    cv::SVDecomp(X, S, U, Vt);

    cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
    for (int i = 0; i < 9; ++i) { H.at<float>(i / 3, i % 3) = Vt.at<float>(8, i); }
    H /= H.at<float>(2, 2);
    return H;
}

void ImageStitcher::_computeInliers(const std::vector<cv::Point2f> &_src,
                                    const std::vector<cv::Point2f> &_dst,
                                    const cv::Mat &_H,
                                    std::vector<int> &_inlierIndices)
{
    if (!_inlierIndices.empty()) { _inlierIndices.clear(); }

    for (size_t i = 0; i < _src.size(); ++i) {
        cv::Mat x = (cv::Mat_<float>(3, 1) << _src[i].x, _src[i].y, 1.f);
        cv::Mat xp = (cv::Mat_<float>(3, 1) << _dst[i].x, _dst[i].y, 1.f);

        cv::Mat Hx = _H * x;
        Hx = Hx / Hx.at<float>(2, 0);

        cv::Mat Hxp = _H.inv() * xp;
        Hxp = Hxp / Hxp.at<float>(2, 0);

        // Use total re-projection error
        // This is L2(xp - Hx) + L2(x - Hxp)
        float error = cv::norm(xp - Hx, cv::NORM_L2) + cv::norm(x - Hxp, cv::NORM_L2);

        if (error < T_DIST) {
            _inlierIndices.push_back(i);
        }
    }
}