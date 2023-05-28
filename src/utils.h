#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>

struct Detection {
    cv::Rect box;
    float conf{};
    int class_id;
};

namespace utils {
    std::size_t VectorProduct(const std::vector<std::int64_t>& vector);

    std::vector<std::string> LoadNames(const std::filesystem::path& path);

    void
    VisualizeDetection(cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string>& class_names);

    void
    LetterBox(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color, bool auto_,
              bool scale_fill, bool scale_up, int stride);

    void ScaleCoords(const cv::Size& image_shape, cv::Rect& box, const cv::Size& image_original_shape);

    void Clip(cv::Rect& rect);

    std::filesystem::path GetFilePathFromRootDir(const std::filesystem::path& filename);
};
