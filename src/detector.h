#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <filesystem>

#include "utils.h"

class YoloDetector {
public:
    YoloDetector(const std::filesystem::path& model_path, const cv::Size& input_size);

    std::vector<Detection> Detect(cv::Mat& image, const float& conf_threshold, const float& iou_threshold);

private:
    void Preprocessing(cv::Mat& image, float*& blob, std::vector<std::int64_t>& input_tensor_shape);
    std::vector<Detection> Postprocessing(const cv::Size& resize_image_shape, const cv::Size& original_image_shape, std::vector<Ort::Value>& output_tensors, const float&conf_threshold, const float& iou_threshold);
    static void GetBestClassInfo(std::vector<float>::iterator it, const int& num_classes, float& best_conf, int& best_class_id);

    Ort::Env env_{ nullptr };
    Ort::SessionOptions session_options_{ nullptr};
    Ort::Session session_{ nullptr};

    std::vector<Ort::AllocatedStringPtr> input_name_allocated_string_{};
    std::vector<Ort::AllocatedStringPtr> output_name_allocated_string_{};
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    bool is_dynamic_input_shape_{};
    cv::Size2f input_image_shape_;
};
