#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "camera.hpp"
#include "light.hpp"
#include "optics_data.hpp"
#include "implicit_geom.hpp"


/**
 * Base class of classifiers 
 */

class Classifier
{
public:
    __host__ __device__ virtual OpticsData transfer(VolumeSampleData v_data, const Camera* cam, Light** lis, int lis_num, float dt) const = 0;
};


/**
 * The classifier for visualizing isosurfaces
 */

class IsosurfaceClassifier : public Classifier
{
protected:
    float isovalue;

public:
    __host__ __device__ IsosurfaceClassifier(float isoval);
    __host__ __device__ virtual OpticsData transfer(VolumeSampleData v_data, const Camera* cam, Light** lis, int lis_num, float dt) const;
};