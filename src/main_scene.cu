#include <cuda.h>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "main_scene.hpp"
#include "utils.hpp"
#include "constant.hpp"
#include "config.hpp"
#include "res_manager.hpp"
#include "constant.hpp"
#include "tinycolormap_cuda.hpp"


extern ResourceManager resource_manager;


__global__ static void createImplicitSurface(ImplicitGeometry** geom, ImplicitGeometry::ImplicitGeometryType geom_type)
{
    SINGLE_THREAD;

    switch (geom_type)
    {
        case ImplicitGeometry::GENUS2:
            *geom = new GenusTwoSurface(Eigen::Vector3f(4.0, 4.0, 4.0));
            break;
        case ImplicitGeometry::WINEGLASS:
            *geom = new WineGlassSurface(Eigen::Vector3f(5.0, 5.0, 6.0));
            break;
        case ImplicitGeometry::POROUS_SURFACE:
            *geom = new PorousSurface(Eigen::Vector3f(2.0, 2.0, 2.0));
            break;
    }
}

__global__ static void createVolume(ImplicitGeometry** geom, VolumeSampleData* vol_data, Eigen::Vector3i grid_dim, Eigen::Vector3f range)
{
    SINGLE_THREAD;

    *geom = new Volume(vol_data, grid_dim, range);
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

__global__ static void createClassifiers(Classifier** cls, int cls_num)
{
    SINGLE_THREAD;

    for (int i = 0; i < cls_num; i++)
    {
        switch ((Classifier::ClassifierType)i)
        {
        case Classifier::ClassifierType::ISOSURFACE:
            cls[i] = new IsosurfaceClassifier;
            break;
        
        case Classifier::ClassifierType::VOLUME:
            cls[i] = new VolumeClassifier;
            break;
        }
    }
}

__global__ static void deleteClassifiers(Classifier** cls, int cls_num)
{
    SINGLE_THREAD;

    for (int i = 0; i < cls_num; ++i)
    {
        delete cls[i];
    }
}

__global__ static void updateClassifier(
    Classifier** cls, Classifier** using_cls, Classifier::ClassifierType select_cls,
    tinycolormap::ColormapType select_cm, Classifier::VisualizationTarget vis_tgt,
    float isovalue, float sigma)
{
    SINGLE_THREAD;

    *using_cls = cls[(int)select_cls];
    (*using_cls)->setColormapType(select_cm);
    (*using_cls)->setVisualizationTarget(vis_tgt);

    if (select_cls == Classifier::ClassifierType::ISOSURFACE)
    {
        IsosurfaceClassifier* isosurf_cls = (IsosurfaceClassifier*)(*using_cls);
        isosurf_cls->setSigma(sigma);
        isosurf_cls->setIsovalue(isovalue);
    }
}


/**
 * MainScene class
 */

MainScene::MainScene()
    : geometry(nullptr)
    , main_camera(nullptr)
    , lights(nullptr)
    , count_lights(0)
    , classifiers(nullptr)
    , using_classifier(nullptr)
    , scene_opened(false)
    , save_next_frame(false)
{
    openScene(DEFAULT_OPEN_GEOM);

    Classifier** d_cls;
    checkCudaErrors(cudaMalloc((void**)&d_cls, sizeof(Classifier*) * NUM_CLASSIFIER_TYPES));
    Classifier** d_using_cls;
    checkCudaErrors(cudaMalloc((void**)&d_using_cls, sizeof(Classifier*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

    createMainCamera<<<1, 1>>>(d_camera, OUTPUT_RESOLUTION_X, OUTPUT_RESOLUTION_Y);
    createClassifiers<<<1, 1>>>(d_cls, NUM_CLASSIFIER_TYPES);
    checkCudaErrors(cudaDeviceSynchronize());

    main_camera = d_camera;
    classifiers = d_cls;
    using_classifier = d_using_cls;
}

MainScene::~MainScene()
{
    closeScene();

    deleteMainCamera<<<1, 1>>>(main_camera);
    deleteClassifiers<<<1, 1>>>(classifiers, NUM_CLASSIFIER_TYPES);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(main_camera);
    cudaFree(classifiers);

    main_camera = nullptr;
    classifiers = nullptr;
    using_classifier = nullptr;
}

void MainScene::updateConfiguration(RenderingConfig* c)
{
    updateMainCamera<<<1, 1>>>(main_camera, c->camera_pos_azimuth, c->camera_pos_polar, c->camera_pos_r, geometry);
    updateLights<<<1, 1>>>(lights, LIGHT_NUM, c->light_rgb, c->light_power);
    updateClassifier<<<1, 1>>>(
        classifiers, using_classifier, c->classifier_type, c->colormap_type, c->visualize_target,
        c->isosurface_classifier_isovalue, c->isosurface_classifier_sigma);
    checkCudaErrors(cudaDeviceSynchronize());
}

void MainScene::closeScene()
{
    deleteImplicitGeometry<<<1, 1>>>(geometry);
    deleteLights<<<1, 1>>>(lights, count_lights);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(geometry);
    cudaFree(lights);

    geometry = nullptr;
    lights = nullptr;
    count_lights = 0;
    scene_opened = false;
}

void MainScene::openScene(const std::string& name, bool builtin)
{
    if (scene_opened)
    {
        closeScene();
    }

    ImplicitGeometry** d_geom;
    checkCudaErrors(cudaMalloc((void**)&d_geom, sizeof(ImplicitGeometry*)));
    Light** d_lis;
    checkCudaErrors(cudaMalloc((void**)&d_lis, sizeof(Light*) * LIGHT_NUM));
    

    if (StrUtils::startsWith(name, BUILTIN_GEOM_GENUS_TWO))
    {
        createImplicitSurface<<<1, 1>>>(d_geom, ImplicitGeometry::GENUS2);
    }
    else if (StrUtils::startsWith(name, BUILTIN_GEOM_WINEGLASS))
    {
        createImplicitSurface<<<1, 1>>>(d_geom, ImplicitGeometry::WINEGLASS);
    }
    else if (StrUtils::startsWith(name, BUILTIN_GEOM_POROUS_SURFACE))
    {
        createImplicitSurface<<<1, 1>>>(d_geom, ImplicitGeometry::POROUS_SURFACE);
    }
    else
    {
        VolumeSampleData* d_vol_data;
        Eigen::Vector3i grid_dim;
        Eigen::Vector3f range;
        std::string path;
        if (builtin)
        {
            path = resource_manager.getResource(name);
        }
        else
        {
            path = name;
        }
        if (Volume::readFromFile(path.c_str(), &d_vol_data, grid_dim, range))
        {
            createVolume<<<1, 1>>>(d_geom, d_vol_data, grid_dim, range);
        }
        else
        {
            UNREACHABLE;
        }
    }
    createLights<<<1, 1>>>(d_lis, LIGHT_NUM, d_geom);
    checkCudaErrors(cudaDeviceSynchronize());

    geometry = d_geom;
    lights = d_lis;
    count_lights = LIGHT_NUM;
    scene_opened = true;
}

void MainScene::saveRenderingResult(const std::string& save_path)
{
    result_save_path = save_path;
    save_next_frame = true;
}

void MainScene::processRenderingResult(unsigned char* bytes)
{
    if (save_next_frame)
    {
        printf("-- Saving the rendering result to %s...\n", result_save_path.c_str());
        if (stbi_write_png(result_save_path.c_str(), OUTPUT_RESOLUTION_X, OUTPUT_RESOLUTION_Y, 4, bytes, 0))
        {
            printf("---- Successfully saved the rendering result to %s\n", result_save_path.c_str());   
        }
        else
        {
            printf("---- Failed to save rendering result to %s\n", result_save_path.c_str());
        }
        save_next_frame = false;
        result_save_path.clear();
    }
}