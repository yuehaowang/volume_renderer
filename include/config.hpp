#pragma once
#include <Eigen/Dense>
#include <imgui.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "tinycolormap_cuda.hpp"
#include "constant.hpp"
#include "classifier.hpp"


/* Rendering output configurations */

#define OUTPUT_RESOLUTION_X 512
#define OUTPUT_RESOLUTION_Y 512
#define OUTPUT_FILE "output.png"


/* Volume data configurations */

#define VOLUME_MAX_DENSITY 9999.0f
#define VOLUME_MIN_DENSITY -9999.0f


/* CUDA configurations */

#define CUDA_BLOCK_THREADS_X 16
#define CUDA_BLOCK_THREADS_Y 16
#define CUDA_DEVICE_ID 0


/* GUI configurations */

#define UI_WINDOW_SIZE_W (OUTPUT_RESOLUTION_X + 400)
#define UI_WINDOW_SIZE_H (OUTPUT_RESOLUTION_Y + 50)
#define UI_WINDOW_TITLE "Volume Renderer"
#define UI_VISUALIZATION_NAME "Visualization"
#define UI_VISUALIZATION_POS ImVec2(0, 0)
#define UI_VISUALIZATION_SIZE ImVec2(OUTPUT_RESOLUTION_X, UI_WINDOW_SIZE_H)
#define UI_CONTROLS_NAME "Controls"
#define UI_CONTROLS_POS ImVec2(UI_VISUALIZATION_SIZE[0], 0)
#define UI_CONTROLS_SIZE ImVec2(UI_WINDOW_SIZE_W - UI_VISUALIZATION_SIZE[0], UI_WINDOW_SIZE_H)


/* Rendering configurations */

#define CAMERA_AZIMUTH_RANGE Eigen::Vector2f(0, 2 * PI)
#define CAMERA_POLAR_RANGE Eigen::Vector2f(0, PI)
#define CAMERA_RADIUS_RANGE Eigen::Vector2f(0.1f, 20.0f)

#define LIGHT_NUM 3
#define LIGHT_POWER_RANGE Eigen::Vector2f(0.05f, 2.0f)

#define SAMPLING_STEP_LEN_RANGE Eigen::Vector2f(0.001f, 0.1f)

#define ISOSURFACE_CLASSIFIER_SIGMA_RANGE Eigen::Vector2f(0.005f, 0.1f)
#define ISOSURFACE_CLASSIFIER_ISOVALUE_RANGE Eigen::Vector2f(-1.0f, 1.0f)

#define AMBIENT_MAGNITUDE 0.1f
#define SPECULAR_SHININESS 16.0f


/**
 * A data structure for storing rendering configurations 
 */

struct RenderingConfig
{
    float camera_pos_azimuth;
    float camera_pos_polar;
    float camera_pos_r;

    Eigen::Vector3f light_rgb;
    float light_power;

    float sampling_step_len;

    Classifier::ClassifierType classifier_type;
    tinycolormap::ColormapType colormap_type;
    Classifier::VisualizationTarget visualize_target;
    float isosurface_classifier_sigma;
    float isosurface_classifier_isovalue;

    int rendering_interval;
};

__global__ void initRenderingConfig(RenderingConfig* render_settings);
