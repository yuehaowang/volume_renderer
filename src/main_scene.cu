#include <cuda.h>
#include <cuda_runtime.h>
#include "main_scene.hpp"
#include "utils.hpp"
#include "constant.hpp"
#include "config.hpp"


__global__ static void createImplicitGeometry(ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    *geom = new PorousSurface(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2.0, 2.0, 2.0));
}

__global__ static void deleteImplicitGeometry(ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    delete *geom;
}

__global__ static void createMainCamera(Camera** cam, int res_x, int res_y)
{
    SINGLE_THREAD;

    Eigen::Vector2i film_res(res_x, res_y);
    *cam = new Camera(Eigen::Vector3f(0, 0, 0), 45.0, film_res);
}

__global__ static void deleteMainCamera(Camera** cam)
{
    SINGLE_THREAD;

    delete *cam;
}

__global__ static void updateMainCamera(Camera** cam, float azimuth, float polar, float r, ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    AABB vol_bbox = (*geom)->getBBox();
    
    r = MathUtils::clamp(r, CAMERA_RADIUS_RANGE[0] + DELTA, CAMERA_RADIUS_RANGE[1] - DELTA) * vol_bbox.diagonalLength() / 2;
    azimuth = MathUtils::clamp(azimuth, CAMERA_AZIMUTH_RANGE[0] + DELTA, CAMERA_AZIMUTH_RANGE[1] - DELTA);
    polar = MathUtils::clamp(polar, CAMERA_POLAR_RANGE[0] + DELTA, CAMERA_POLAR_RANGE[1] - DELTA);
    Eigen::Vector3f offset(r * cos(azimuth) * sin(polar), r * cos(polar), r * sin(azimuth) * sin(polar));

    Eigen::Vector3f camera_pos = vol_bbox.getCenter() + offset;
    Eigen::Vector3f camera_look_at = vol_bbox.getCenter();

    Eigen::Vector3f ref_up;
    if (polar < PI / 2)
    {
        ref_up = Eigen::Vector3f(0, 1, 0);
    }
    else
    {
        ref_up = Eigen::Vector3f(0, 1, 0);
    }
    
    (*cam)->moveTo(camera_pos);
    (*cam)->lookAt(camera_look_at, ref_up);
}

__global__ static void createLights(Light** lis, int lis_num, ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    AABB vol_bbox = (*geom)->getBBox();

    /* Centric light */
    lis[0] = new Light(vol_bbox.getCenter(), Eigen::Vector3f(1.0, 1.0, 1.0));
    
    /* Surrounding lights */
    int sur_lis_num = lis_num - 1;
    float r = vol_bbox.diagonalLength() / 2;
    for (int i = 0; i < sur_lis_num; ++i)
    {
        float theta = i * (2 * PI / sur_lis_num);
        float phi = PI / 4;

        Eigen::Vector3f p(r * cos(phi) * cos(theta), r * sin(phi), r * cos(phi) * sin(theta));
        lis[i + 1] = new Light(p, Eigen::Vector3f(0, 0, 0));
    }
}

__global__ static void deleteLights(Light** lis, int lis_num)
{
    SINGLE_THREAD;

    for (int i = 0; i < lis_num; ++i)
    {
        delete lis[i];
    }
}

__global__ static void updateLights(Light** lis, int lis_num, Eigen::Vector3f rgb, float power)
{
    SINGLE_THREAD;

    for (int i = 0; i < lis_num; ++i)
    {
        lis[i]->setRadiance(rgb * power);
    }
}

__global__ static void createClassifier(Classifier** cls)
{
    SINGLE_THREAD;

    *cls = new IsosurfaceClassifier(0.0);
}

__global__ static void deleteClassifier(Classifier** cls)
{
    SINGLE_THREAD;

    delete *cls;
}


/**
 * MainScene class
 */

MainScene::MainScene()
{
    int res_x = RESOLUTION_X;
    int res_y = RESOLUTION_Y;

    ImplicitGeometry** d_geom;
    checkCudaErrors(cudaMalloc((void**)&d_geom, sizeof(ImplicitGeometry*)));
    Light** d_lis;
    checkCudaErrors(cudaMalloc((void**)&d_lis, sizeof(Light*) * LIGHT_NUM));
    Classifier** d_cls;
    checkCudaErrors(cudaMalloc((void**)&d_cls, sizeof(Classifier*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

    createImplicitGeometry<<<1, 1>>>(d_geom);
    createMainCamera<<<1, 1>>>(d_camera, res_x, res_y);
    createLights<<<1, 1>>>(d_lis, LIGHT_NUM, d_geom);
    createClassifier<<<1, 1>>>(d_cls);
    checkCudaErrors(cudaDeviceSynchronize());

    geometry = d_geom;
    main_camera = d_camera;
    classifier = d_cls;
    lights = d_lis;
    count_lights = LIGHT_NUM;
}

MainScene::~MainScene()
{
    deleteImplicitGeometry<<<1, 1>>>(geometry);
    deleteMainCamera<<<1, 1>>>(main_camera);
    deleteLights<<<1, 1>>>(lights, count_lights);
    deleteClassifier<<<1, 1>>>(classifier);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(geometry);
    cudaFree(main_camera);
    cudaFree(classifier);
    cudaFree(lights);
}

void MainScene::updateConfiguration(RenderingConfig* c)
{
    updateMainCamera<<<1, 1>>>(main_camera, c->camera_pos_azimuth, c->camera_pos_polar, c->camera_pos_r, geometry);
    updateLights<<<1, 1>>>(lights, LIGHT_NUM, c->light_rgb, c->light_power);
    checkCudaErrors(cudaDeviceSynchronize());
}