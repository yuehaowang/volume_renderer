#pragma once
#include <glad/glad.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "camera.hpp"
#include "classifier.hpp"
#include "light.hpp"
#include "implicit_geom.hpp"


/**
 * Renderer for volume data
 */

class VolumeRenderer
{
protected:
    Camera** main_camera;
    Light** lights;
    int count_lights;
    ImplicitGeometry** geometry;
    Classifier** classifier;

public:
    VolumeRenderer();
    virtual ~VolumeRenderer();
    void setCamera(Camera** cam);
    void setLights(Light** lis, int lis_num);
    void setVolume(ImplicitGeometry** geom);
    void setClassifier(Classifier** cls);
    void renderFrontToBack(Eigen::Vector3f* pixel_array, int res_x, int res_y, float dt);
};
