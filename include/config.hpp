#pragma once
#include <Eigen/Dense>
#include <imgui.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "constant.hpp"


/* Constant settings */

#define AMBIENT_MAGNITUDE 0.1f
#define SPECULAR_SHININESS 16.0f

#define LIGHT_NUM 3

#define SAMPLE_STEP_LENGTH 0.008
#define RESOLUTION_X 512
#define RESOLUTION_Y 512
#define OUTPUT_FILE "output.png"

#define UI_WINDOW_SIZE_W (RESOLUTION_X + 400)
#define UI_WINDOW_SIZE_H (RESOLUTION_Y + 50)
#define UI_WINDOW_TITLE "Volume Renderer"
#define UI_VISUALIZATION_NAME "Visualization"
#define UI_VISUALIZATION_POS ImVec2(0, 0)
#define UI_VISUALIZATION_SIZE ImVec2(RESOLUTION_X, UI_WINDOW_SIZE_H)
#define UI_CONTROLS_NAME "Controls"
#define UI_CONTROLS_POS ImVec2(UI_VISUALIZATION_SIZE[0], 0)
#define UI_CONTROLS_SIZE ImVec2(UI_WINDOW_SIZE_W - UI_VISUALIZATION_SIZE[0], UI_WINDOW_SIZE_H)


/* Rendering configurations */

#define CAMERA_AZIMUTH_RANGE Eigen::Vector2f(0, 2 * PI)
#define CAMERA_POLAR_RANGE Eigen::Vector2f(0, PI)
#define CAMERA_RADIUS_RANGE Eigen::Vector2f(0.1f, 20.0f)

#define LIGHT_POWER_RANGE Eigen::Vector2f(1.0f, 5000.0f)

#define SAMPLING_STEP_LEN_RANGE Eigen::Vector2f(0.001f, 0.1f)

struct RenderingConfig
{
    float camera_pos_azimuth;
    float camera_pos_polar;
    float camera_pos_r;

    Eigen::Vector3f light_rgb;
    float light_power;

    float sampling_step_len;
};

__global__ void init_rendering_config(RenderingConfig* render_settings);

