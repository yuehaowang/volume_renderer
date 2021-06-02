#include "config.hpp"
#include "utils.hpp"


__global__ void init_rendering_config(RenderingConfig* render_settings)
{
    SINGLE_THREAD;

    /* Set default values */
    render_settings->camera_pos_azimuth = PI / 4;
    render_settings->camera_pos_polar = PI / 4;
    render_settings->camera_pos_r = 3;
}
