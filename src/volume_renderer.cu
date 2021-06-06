#include "volume_renderer.hpp"
#include "optics_data.hpp"
#include "config.hpp"
#include "utils.hpp"


/**
 * Kernels for ray casting and image composition
 */

__device__ static void compositeFrontToBack(
    Eigen::Vector3f& color_dst, Eigen::Vector3f& alpha_dst,
    Eigen::Vector3f color_src, Eigen::Vector3f alpha_src)
{
    color_dst = color_dst + (Eigen::Vector3f::Ones() - alpha_dst).cwiseProduct(color_src);
    alpha_dst = (alpha_dst + (Eigen::Vector3f::Ones() - alpha_dst).cwiseProduct(alpha_src)).cwiseMin(1.0).cwiseMax(0.0);
}

__global__ static void rayIntegral(
    Eigen::Vector3f* pixel_array, ImplicitGeometry** geom, Camera** cam,
    Light** lis, int lis_num, Classifier** cls, float ambient, float shininess, float dt)
{
    int max_x = (*cam)->getFilm().resolution.x();
    int max_y = (*cam)->getFilm().resolution.y();

    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if((dx >= max_x) || (dy >= max_y))
    {
        return;
    }

    Eigen::Vector3f color(0, 0, 0);
    Eigen::Vector3f alpha(0, 0, 0);
    Ray ray = (*cam)->generateRay((float)dx, (float)dy);
    float t_start = 0, t_end = 0;

    /* Integration calculation */
    if ((*geom)->bboxRayIntersection(ray, t_start, t_end))
    {
        float t = t_start;
        while (t <= t_end)
        {
            /* Get position of the sampled point */
            Eigen::Vector3f p = ray.getPoint(t);
            /* Sample the point on the geometry */
            VolumeSampleData pt_data = (*geom)->sample(p);

            /* Get optical data by transfer function */
            OpticsData opt_d = (*cls)->transfer(pt_data, *cam, lis, lis_num, ambient, shininess, dt);

            /* Front-to-back composition */
            compositeFrontToBack(color, alpha, opt_d.getColor(), opt_d.getOpacity());

            /* Early ray termination */
            if (alpha.x() >= 1.0 && alpha.y() >= 1.0 && alpha.z() >= 1.0)
            {
                break;
            }

            t += dt;
        }
    }
    pixel_array[dy * max_x + dx] = color;
}


/**
 * VolumeRenderer class
 */

VolumeRenderer::VolumeRenderer()
    : main_camera(nullptr)
    , classifier(nullptr)
    , geometry(nullptr)
    , lights(nullptr)
{
}

VolumeRenderer::~VolumeRenderer()
{
}

void VolumeRenderer::setCamera(Camera** cam)
{
    main_camera = cam;
}

void VolumeRenderer::setLights(Light** lis, int lis_num)
{
    lights = lis;
    count_lights = lis_num;
}

void VolumeRenderer::setVolume(ImplicitGeometry** geom)
{
    geometry = geom;
}

void VolumeRenderer::setClassifier(Classifier** cls)
{
    classifier = cls;

}

void VolumeRenderer::renderFrontToBack(Eigen::Vector3f* pixel_array, int res_x, int res_y, float ambient, float shininess, float dt)
{
    int tx = CUDA_BLOCK_THREADS_X;
    int ty = CUDA_BLOCK_THREADS_Y;
    dim3 blocks(res_x / tx + 1, res_y / ty + 1);
    dim3 threads(tx, ty);
    rayIntegral<<<blocks, threads>>>(pixel_array, geometry, main_camera, lights, count_lights, classifier, ambient, shininess, dt);
    checkCudaErrors(cudaDeviceSynchronize());
}
