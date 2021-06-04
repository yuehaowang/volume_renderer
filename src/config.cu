#include "config.hpp"
#include "utils.hpp"


__global__ void initRenderingConfig(RenderingConfig* render_settings)
{
    SINGLE_THREAD;

    /* Set default values */

    render_settings->camera_pos_azimuth = PI / 4;
    render_settings->camera_pos_polar = PI / 4;
    render_settings->camera_pos_r = 3;

    render_settings->light_power = 1000;
    render_settings->light_rgb = Eigen::Vector3f(1, 1, 1);

    render_settings->sampling_step_len = 0.008;

    render_settings->rendering_interval = 3;
}
