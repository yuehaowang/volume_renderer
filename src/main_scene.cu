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

__global__ static void create_main_camera(Camera** cam, int res_x, int res_y, ImplicitGeometry** geom)
{
    SINGLE_THREAD;

    AABB vol_bbox = (*geom)->getBBox();
    Eigen::Vector3f camera_pos = vol_bbox.getCenter() + vol_bbox.getSize() * 1.5;
    Eigen::Vector3f camera_look_at = vol_bbox.getCenter();
    Eigen::Vector2i film_res(res_x, res_y);
    
    *cam = new Camera(camera_pos, 45.0, film_res);
    (*cam)->lookAt(camera_look_at, Eigen::Vector3f(0, 0, 1));
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
        lis[i + 1] = new Light(p, Eigen::Vector3f(1.0, 1.0, 1.0));
    }
}

__global__ static void create_classifier(Classifier** cls)
{
    SINGLE_THREAD;

    *cls = new IsosurfaceClassifier(0.0);
}

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
    create_main_camera<<<1, 1>>>(d_camera, res_x, res_y, d_geom);
    create_lights<<<1, 1>>>(d_lis, LIGHT_NUM, d_geom);
    create_classifier<<<1, 1>>>(d_cls);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    main_camera = d_camera;
    geometry = d_geom;
    classifier = d_cls;
    lights = d_lis;
}
