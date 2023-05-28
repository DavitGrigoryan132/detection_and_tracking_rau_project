cmake_minimum_required(VERSION 3.21)
project(dg_cv_viewer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED)
find_package(OnnxRuntime REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(dg_cv_viewer
        src/main.cpp
        src/camera.cpp
        src/detector.cpp
        src/utils.cpp
        src/main_window.cpp
        src/tracker.cpp)

target_link_libraries(dg_cv_viewer
        qt::qt
        onnxruntime::onnxruntime
        opencv::opencv
        )

