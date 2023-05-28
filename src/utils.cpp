#include "utils.h"

#include <fstream>

std::size_t utils::VectorProduct(const std::vector<std::int64_t>& vector) {
    if (vector.empty()) {
        return 0;
    }

    std::size_t product = 1;
    for (const auto& number: vector) {
        product *= number;
    }

    return product;
}

std::vector<std::string> utils::LoadNames(const std::filesystem::path& path) {
    std::vector<std::string> class_names;
    std::ifstream input_file(path);

    if (input_file.good()) {
        std::string line;
        while (getline(input_file, line)) {
            if (line.back() == '\r') {
                line.pop_back();
            }
            class_names.push_back(line);
        }
        input_file.close();
    }
    else {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return class_names;
}

void utils::VisualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
                               const std::vector<std::string>& class_names) {
    for (const auto& detection: detections) {
        cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

        int x = detection.box.x;
        int y = detection.box.y;

        auto conf = static_cast<int>(std::round(detection.conf * 100));
        int class_id = detection.class_id;
        std::string label = class_names[class_id] + " " + std::to_string(conf);

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
        cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + size.width, y), cv::Scalar(229, 160, 21), -1);

        cv::putText(image, label, cv::Point(x, y - 3), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
    }
}

void utils::LetterBox(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape, const cv::Scalar& color,
                      bool auto_, bool scale_fill, bool scale_up, int stride) {
    cv::Size shape = image.size();
    float r = std::min(static_cast<float>(new_shape.height) / static_cast<float>(shape.height),
                       static_cast<float>(new_shape.width) / static_cast<float>(shape.width));

    if (!scale_up) {
        r = std::min(r, 1.f);
    }

    std::array<int, 2> new_unpad{ static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                                  static_cast<int>(std::round(static_cast<float>(shape.height) * r)) };

    auto dw = static_cast<float>(new_shape.width - new_unpad[0]);
    auto dh = static_cast<float>(new_shape.height - new_unpad[1]);

    if (auto_) {
        dw = static_cast<float>(static_cast<int>(dw) % stride);
        dh = static_cast<float>(static_cast<int>(dh) % stride);
    }
    else if (scale_fill) {
        dw = 0.f;
        dh = 0.f;
        new_unpad[0] = new_shape.width;
        new_unpad[1] = new_shape.height;
    }

    dw /= 2.f;
    dh /= 2.f;

    if (shape.width != new_unpad[0] and shape.height != new_unpad[1]) {
        cv::resize(image, out_image, cv::Size(new_unpad[0], new_unpad[1]));
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::ScaleCoords(const cv::Size& image_shape, cv::Rect& box, const cv::Size& image_original_shape) {
    float gain = std::min(static_cast<float>(image_shape.height) / static_cast<float>(image_original_shape.height),
                          static_cast<float>(image_shape.width) / static_cast<float>(image_original_shape.width));

    std::array<int, 2> pad{ static_cast<int>((static_cast<float>(image_shape.width) -
                                              static_cast<float>(image_original_shape.width) * gain) / 2.f),
                            static_cast<int>((static_cast<float>(image_shape.height) -
                                              static_cast<float>(image_original_shape.width) * gain) / 2.f) };

    box.x = static_cast<int>(std::round(static_cast<float>(box.x - pad[0]) / gain));
    box.y = static_cast<int>(std::round(static_cast<float>(box.y - pad[1]) / gain));

    box.width = static_cast<int>(std::round(static_cast<float>(box.width) / gain));
    box.height = static_cast<int>(std::round(static_cast<float>(box.height) / gain));
}

void utils::Clip(cv::Rect& rect) {
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    if (rect.y + rect.height >= 640) {
        rect.height = 639 - rect.y;
    }
    if (rect.x + rect.width >= 640) {
        rect.width = 639 - rect.x;
    }
}

std::filesystem::path utils::GetFilePathFromRootDir(const std::filesystem::path& filename) {
    auto root_dir = std::getenv("ROOT_DIR");

    return std::filesystem::path(root_dir) / filename;
}
