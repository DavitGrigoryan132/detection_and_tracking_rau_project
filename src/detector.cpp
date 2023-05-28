#include "detector.h"

YoloDetector::YoloDetector(const std::filesystem::path& model_path, const cv::Size& input_size) {
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "ONNX_DETECTION");
    session_options_ = Ort::SessionOptions();

    std::vector<std::string> available_providers = Ort::GetAvailableProviders();

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    std::vector<std::int64_t> input_tensor_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

    is_dynamic_input_shape_ = false;
    if (input_tensor_shape[2] == -1 and input_tensor_shape[3] == -1) {
        std::cout << "Dynamic input shape" << std::endl;
        is_dynamic_input_shape_ = true;
    }

    for (const auto& shape: input_tensor_shape) {
        std::cout << "Input shape: " << shape << std::endl;
    }

    input_name_allocated_string_.push_back(std::move(session_.GetInputNameAllocated(0, allocator)));
    output_name_allocated_string_.push_back(std::move(session_.GetOutputNameAllocated(0, allocator)));

    input_names_.push_back(input_name_allocated_string_.back().get());
    output_names_.push_back(output_name_allocated_string_.back().get());

    input_image_shape_ = cv::Size2f(input_size);
}

void YoloDetector::GetBestClassInfo(std::vector<float>::iterator it, const int& num_classes, float& best_conf,
                                    int& best_class_id) {
    best_class_id = 5;
    best_conf = 0;

    for (int i = 5; i < num_classes + 5; ++i) {
        if (it[i] > best_conf) {
            best_conf = it[i];
            best_class_id = i - 5;
        }
    }
}

void YoloDetector::Preprocessing(cv::Mat& image, float*& blob, std::vector<std::int64_t>& input_tensor_shape) {
    cv::Mat resized_image;
    cv::Mat float_image;

    cv::cvtColor(image, resized_image, cv::COLOR_BGR2RGB);
    utils::LetterBox(resized_image, resized_image, input_image_shape_, cv::Scalar(114, 114, 114),
                     is_dynamic_input_shape_, false, true, 32);

    input_tensor_shape[2] = resized_image.rows;
    input_tensor_shape[3] = resized_image.cols;

    resized_image.convertTo(float_image, CV_32FC3, 1 / 255.);
    blob = new float[float_image.cols * float_image.rows * float_image.channels()];
    cv::Size float_image_size{ float_image.cols, float_image.rows };

    std::vector<cv::Mat> chw(float_image.channels());
    for (int i = 0; i < float_image.channels(); ++i) {
        chw[i] = cv::Mat(float_image_size, CV_32FC1, blob + i * float_image_size.width * float_image_size.height);
    }
    cv::split(float_image, chw);
}

std::vector<Detection>
YoloDetector::Postprocessing(const cv::Size& resize_image_shape, const cv::Size& original_image_shape,
                             std::vector<Ort::Value>& output_tensors, const float& conf_threshold,
                             const float& iou_threshold) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    auto* raw_output = output_tensors[0].GetTensorData<float>();
    std::vector<std::int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(raw_output, raw_output + count);

    auto num_classes = static_cast<int>(output_shape[2] - 5);
    auto elements_in_batch = static_cast<int>(output_shape[1] * output_shape[2]);

    for (auto it = output.begin(); it != output.begin() + elements_in_batch; it += output_shape[2]) {
        float class_confidence = it[4];

        if (class_confidence > conf_threshold) {
            auto center_x = static_cast<int>(it[0]);
            auto center_y = static_cast<int>(it[1]);
            auto width = static_cast<int>(it[2]);
            auto height = static_cast<int>(it[3]);
            int left = center_x - width / 2;
            int top = center_y - height / 2;

            float obj_conf;
            int class_id;
            GetBestClassInfo(it, num_classes, obj_conf, class_id);

            float confidence = class_confidence * obj_conf;

            boxes.emplace_back(left, top, width, height);
            confidences.emplace_back(confidence);
            class_ids.emplace_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    std::vector<Detection> detections;

    for (int idx: indices) {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::ScaleCoords(resize_image_shape, det.box, original_image_shape);

        det.conf = confidences[idx];
        det.class_id = class_ids[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YoloDetector::Detect(cv::Mat& image, const float& conf_threshold, const float& iou_threshold) {
    float* blob = nullptr;
    std::vector<std::int64_t> input_tensor_shape{ 1, 3, -1, -1 };
    Preprocessing(image, blob, input_tensor_shape);

    std::size_t input_tensor_size = utils::VectorProduct(input_tensor_shape);
    std::vector<float> input_tensor_values(blob, blob + input_tensor_size);

    std::vector<Ort::Value> input_tensors;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                            input_tensor_shape.data(), input_tensor_shape.size()));

    std::vector<Ort::Value> output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(),
                                                          input_tensors.data(), 1, output_names_.data(), 1);

    cv::Size resized_shape = cv::Size(static_cast<int>(input_tensor_shape[3]), static_cast<int>(input_tensor_shape[2]));
    std::vector<Detection> result = Postprocessing(resized_shape, image.size(), output_tensors, conf_threshold,
                                                   iou_threshold);

    delete[] blob;

    return result;
}
