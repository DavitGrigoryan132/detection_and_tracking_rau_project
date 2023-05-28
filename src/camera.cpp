#include "camera.h"

#include <iostream>
#include <QActionGroup>
#include <QVariant>
#include <QWidget>
#include <QKeyEvent>
#include <opencv2/imgproc.hpp>


Camera::Camera(QWidget* parent) : QWidget(parent), cap_(0),
                                  detector_{ utils::GetFilePathFromRootDir("yolov5s.onnx"),
                                             cv::Size(640, 640) }, class_names_{
                utils::LoadNames(utils::GetFilePathFromRootDir("coco.names")) } {
    setFixedSize(1280, 720);

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    timer_ = new QTimer(this);
    label_.resize(1280, 720);
    label_.setParent(this);
    connect(timer_, SIGNAL(timeout()), this, SLOT(updatePicture()));
    timer_->start(20);
}

void Camera::updatePicture() {
    if (cap_.isOpened()) {
        frames_count_++;
        cv::Mat image;

        cap_ >> image;

        // TODO fix bug of rescaling
        cv::resize(image, image, cv::Size(640, 640));
        if (frames_count_ % 20 != 0 and detected_) {
            std::cout << "Tracking\n";
            for (std::size_t i = 0; i < detections_.size(); ++i) {
                try {
                    trackers_[i].Track(image, prev_frame_, detections_[i].box);
                    utils::Clip(detections_[i].box);
                }
                catch (const std::runtime_error& e) {
                    std::cerr << e.what();
                    detections_.erase(detections_.begin() + static_cast<long>(i));
                }
            }

            utils::VisualizeDetection(image, detections_, class_names_);
        }
        else {
            trackers_.clear();
            std::cout << "Detection\n";
            detections_ = detector_.Detect(image, 0.3f, 0.4f);
            detected_ = not detections_.empty();
            for (auto& detection: detections_) {
                utils::Clip(detection.box);
                trackers_.emplace_back(image, detection.box);
            }
        }

        prev_frame_ = image;

        cv::resize(image, image, cv::Size(1280, 720));

        //conversion from Mat to QImage
        cv::Mat dest;
        cv::cvtColor(image, dest, cv::COLOR_BGR2RGB);
        QImage qimage = QImage((uchar*) dest.data, dest.cols, dest.rows, static_cast<int>(dest.step),
                               QImage::Format_RGB888);

        //show Qimage using QLabel
        label_.setPixmap(QPixmap::fromImage(qimage));
    }
    else {
        frames_count_ = 0;

        cv::Mat image = cv::Mat::zeros(720, 1280, CV_8UC3);
        QImage qimage = QImage((uchar*) image.data, image.cols, image.rows, static_cast<int>(image.step),
                               QImage::Format_RGB888);

        label_.setPixmap(QPixmap::fromImage(qimage));
    }
}

void Camera::SetCapIndex(int index) {
    cap_.open(index);
}

Camera::~Camera() {
    cap_.release();
}
