#include "main_window.h"

#include <QResizeEvent>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/registry.hpp>

MainWindow::MainWindow(): QMainWindow(), layout_{this}, camera_{ this }, combo_box_{this} {

    layout_.addWidget(&camera_);

    layout_.addWidget(&combo_box_);

    setLayout(&layout_);
    auto backends = cv::videoio_registry::getCameraBackends();

    int index = 0;
    // Print list of backends and their indexes
    for (auto& backend : backends) {
        combo_box_.addItem(cv::videoio_registry::getBackendName(backend).c_str());
        std::cout << "Backend " << cv::videoio_registry::getBackendName(backend) << " has index " << index++ << std::endl;
    }
    setMinimumSize(layout_.minimumSize());
    connect(&combo_box_, SIGNAL(currentIndexChanged(int)), &camera_, SLOT(SetCapIndex(int)));
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    auto parent_center = rect().center();

    // get the child widget's size and center point
    auto child_center = camera_.rect().center();

    // calculate the position of the child widget to center it in the parent widget
    int x = parent_center.x() - child_center.x();
    int y = parent_center.y() - child_center.y();

    // set the geometry of the child widget to position it in the center of the parent widget
    camera_.move(x, y);
}


