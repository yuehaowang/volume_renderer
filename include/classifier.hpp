#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#define TINYCOLORMAP_WITH_EIGEN
#include "tinycolormap_cuda.hpp"
#include "camera.hpp"
#include "light.hpp"
#include "optics_data.hpp"
#include "voxel.hpp"


#define NUM_CLASSIFIER_TYPES 2

/**
 * Base class of classifiers 
 */

class Classifier
{
public:
    enum ClassifierType
    {
        ISOSURFACE,
        VOLUME
    };

    enum VisualizationTarget
    {
        NORMAL_X,
        NORMAL_Y,
        NORMAL_Z,
        VALUE
    };

protected:
    float kappa;
    VisualizationTarget visualize_target;
    tinycolormap::ColormapType cm_type;

public:
    __host__ __device__ Classifier(float K = 0.1f, VisualizationTarget vis_tgt = VisualizationTarget::VALUE, tinycolormap::ColormapType cm = tinycolormap::ColormapType::Heat);
    __host__ __device__ virtual Eigen::Vector3f getMappedColor(float value, const Eigen::Vector3f& normal) const;
    __host__ __device__ virtual float transfer(const VolumeSampleData& v_data) const = 0;
    __host__ __device__ virtual OpticsData transfer(const VolumeSampleData& v_data, const Camera* cam, Light** lis, int lis_num, float dt) const = 0;
};


/**
 * The classifier for visualizing isosurfaces
 */

class IsosurfaceClassifier : public Classifier
{
protected:
    float isovalue;
    float sigma;

public:
    __host__ __device__ IsosurfaceClassifier();
    __host__ __device__ virtual float transfer(const VolumeSampleData& v_data) const;
    __host__ __device__ virtual OpticsData transfer(const VolumeSampleData& v_data, const Camera* cam, Light** lis, int lis_num, float dt) const;
};


/**
 * The classifier for visualizing entire volume
 */

class VolumeClassifier : public Classifier
{
protected:
    float min_transparency;

public:
    __host__ __device__ VolumeClassifier();
    __host__ __device__ virtual float transfer(const VolumeSampleData& v_data) const;
    __host__ __device__ virtual OpticsData transfer(const VolumeSampleData& v_data, const Camera* cam, Light** lis, int lis_num, float dt) const;
};