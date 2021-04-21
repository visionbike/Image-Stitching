#include "feature.hpp"

void detectSIFTFeature(const cv::Mat &_img, std::vector<cv::KeyPoint> &_kpts, cv::Mat &_desc)
{
    // Detect SIFT key points
    std::vector<cv::KeyPoint> kpts;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->detectAndCompute(_img, cv::Mat(), _kpts, _desc);
}

void matchFeature(const cv::Mat &_desc1, const cv::Mat &_desc2, std::vector<cv::DMatch> &_matches) {
    _matches.clear();

    // Apply bruce force matching
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(_desc1, _desc2, _matches, cv::Mat());
}

void keepGoodMatches(const std::vector<cv::DMatch> &_matches, std::vector<cv::DMatch> &_gmatches)
{
    _gmatches.clear();

    // Calculate min-max distances between key points
    float minDist = 100.f;
    float maxDist = 0.f;
    for (auto & _match : _matches) {
        if (_match.distance < minDist) { minDist = _match.distance; }
        if (_match.distance > maxDist) { maxDist = _match.distance; }
    }

    std::cout << "[INFO] Min dist: " << minDist << std::endl;
    std::cout << "[INFO] Max dist: " << maxDist << std::endl;

    // Use only "good" matches whose distance is less than 3x minDist
    for (auto & _match : _matches) {
        if (_match.distance <  2 * minDist) { _gmatches.push_back(_match); }
    }
}
