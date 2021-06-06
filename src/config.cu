#include "config.hpp"
#include "utils.hpp"


__global__ void initRenderingConfig(RenderingConfig* render_settings)
{
    SINGLE_THREAD;

    /* Set default values */

    render_settings->camera_pos_azimuth = PI / 4;
    render_settings->camera_pos_polar = PI / 4;
    render_settings->camera_pos_r = 3;

    render_settings->light_power = 1.0f;
    render_settings->light_rgb = Eigen::Vector3f(1, 1, 1);

    render_settings->sampling_step_len = 0.008;

    render_settings->classifier_type = Classifier::ClassifierType::ISOSURFACE;
    render_settings->colormap_type = tinycolormap::ColormapType::Heat;
    render_settings->visualize_target = Classifier::VisualizationTarget::VALUE;
    render_settings->isosurface_classifier_isovalue = 0;
    render_settings->isosurface_classifier_sigma = 0.01;

    render_settings->rendering_interval = 3;
}
