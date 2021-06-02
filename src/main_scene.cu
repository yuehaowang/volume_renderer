#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "main_scene.hpp"
#include "utils.hpp"
#include "constant.hpp"
#include "config.hpp"


__global__ static void create_implicit_geometry(ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    *geom = new PorousSurface(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2.0, 2.0, 2.0));
}

__global__ static void create_main_camera(Camera** cam, int res_x, int res_y)
{
    SINGLE_THREAD;

    Eigen::Vector2i film_res(res_x, res_y);
    *cam = new Camera(Eigen::Vector3f(0, 0, 0), 45.0, film_res);
}

__global__ static void update_main_camera(Camera** cam, float azimuth, float polar, float r, ImplicitGeometry** geom)
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

__global__ static void create_lights(Light** lis, int lis_num, ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    AABB vol_bbox = (*geom)->getBBox();

    /* Centeric light */
    lis[0] = new Light(vol_bbox.getCenter(), Eigen::Vector3f(1.0, 1.0, 1.0));
    
    /* Surrounding lights */
    int sur_lis_num = lis_num - 1;
    float r = vol_bbox.diagonalLength() / 2;
    for (int i = 0; i < sur_lis_num; ++i)
    {
        float theta = i * (2 * PI / sur_lis_num);
        float phi = PI / 4;

        Eigen::Vector3f p(r * cos(phi) * cos(theta), r * cos(phi) * sin(theta), r * sin(phi));
        lis[i + 1] = new Light(p, LIGHT_POWER);
    }
}

__global__ static void create_classifier(Classifier** cls)
{
    SINGLE_THREAD;

    *cls = new IsosurfaceClassifier(0.0);
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

    create_implicit_geometry<<<1, 1>>>(d_geom);
    create_main_camera<<<1, 1>>>(d_camera, res_x, res_y);
    create_lights<<<1, 1>>>(d_lis, LIGHT_NUM, d_geom);
    create_classifier<<<1, 1>>>(d_cls);
    checkCudaErrors(cudaDeviceSynchronize());

    main_camera = d_camera;
    geometry = d_geom;
    classifier = d_cls;
    lights = d_lis;
}


void MainScene::updateConfiguration(RenderingConfig* c)
{
    update_main_camera<<<1, 1>>>(main_camera, c->camera_pos_azimuth, c->camera_pos_polar, c->camera_pos_r, geometry);
    checkCudaErrors(cudaDeviceSynchronize());
}