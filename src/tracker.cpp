#include "tracker.h"

Tracker::Tracker(const cv::Mat& frame, const cv::Rect& bbox) {
    cv::Mat box_image;
    cv::cvtColor(frame(bbox), box_image, cv::COLOR_BGR2GRAY);

    cv::goodFeaturesToTrack(box_image, prev_points_, 100, 0.3, 1);
}

void Tracker::Track(const cv::Mat& frame, const cv::Mat& prev_frame, cv::Rect& bbox) {
    std::vector<cv::Point2f> current_points;
    std::vector<std::uint8_t> status;
    std::vector<float> err;
    cv::TermCriteria term_criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    if (prev_points_.empty()) {
        throw std::runtime_error("No prev points");
    }

    cv::calcOpticalFlowPyrLK(prev_frame, frame, prev_points_, current_points, status, err, cv::Size(15, 15), 2,
                             term_criteria);

    std::vector<cv::Point2f> good_points;
    for (std::size_t i = 0; i < current_points.size(); ++i) {
        if (status[i]) {
            good_points.push_back(current_points[i]);
        }
    }

    if (good_points.empty()) {
        throw std::runtime_error("No good points");
    }

    bbox.x += static_cast<int>(Mean(current_points, 0) - Mean(prev_points_, 0));
    bbox.y += static_cast<int>(Mean(current_points, 1) - Mean(prev_points_, 1));

    prev_points_ = current_points;
}

float Tracker::Mean(const std::vector<cv::Point2f>& input_array, const int axis) {
    std::vector<float> data(input_array.size());
    for (const auto& point: input_array) {
        if (axis == 0) {
            data.push_back(point.x);
        }
        else if (axis == 1) {
            data.push_back(point.y);
        }
    }

    return std::reduce(data.begin(), data.end()) / static_cast<float>(data.size());
}

