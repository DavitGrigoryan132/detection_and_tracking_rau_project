#pragma once

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

class Tracker {
public:
    Tracker(const cv::Mat& frame, const cv::Rect& bbox);

    void Track(const cv::Mat& frame, const cv::Mat& prev_frame, cv::Rect& bbox);

private:
    static float Mean(const std::vector<cv::Point2f>& input_array, int axis);

    std::vector<cv::Point2f> prev_points_;
};
