#pragma once

#include "camera.h"

#include <QMainWindow>
#include <QVBoxLayout>
#include <QComboBox>

class MainWindow: public QMainWindow {
public:
    MainWindow();

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    Camera camera_;
    QComboBox combo_box_;
    QVBoxLayout layout_;
};
