//
// Created by davit on 06.04.23.
//

#pragma once

#include <QTimer>
#include <QLabel>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <QMainWindow>
#include <QDebug>
#include <QWidget>

#include "detector.h"
#include "tracker.h"
#include "utils.h"

class Camera : public QWidget {
Q_OBJECT
public:
    explicit Camera(QWidget* parent = nullptr);

    ~Camera() override;

public slots:
    void SetCapIndex(int index);
    void updatePicture();

private:
    YoloDetector detector_;
    std::vector<std::string> class_names_;
    QTimer* timer_;
    QLabel label_;
    cv::VideoCapture cap_;
    int frames_count_{ 0 };
    std::vector<Detection> detections_{};
    std::vector<Tracker> trackers_;
    bool detected_ = false;
    cv::Mat prev_frame_{};
};
