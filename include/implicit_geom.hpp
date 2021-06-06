#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "bbox.hpp"
#include "ray.hpp"
#include "voxel.hpp"


/**
 * Base class of implicit geometries
 */

class ImplicitGeometry
{
protected:
    AABB bounding_box;

public:
    enum ImplicitGeometryType
    {
        GENUS2,
        WINEGLASS,
        POROUS_SURFACE,
        VOLUME_FILE
    };

    __host__ __device__ ImplicitGeometry(const Eigen::Vector3f& range);
    __host__ __device__ AABB getBBox() const;
    __host__ __device__ virtual const VolumeSampleData& sample(const Eigen::Vector3f& p) = 0;
    __host__ __device__ virtual bool bboxRayIntersection(const Ray& r, float& t_entry, float& t_exit) const;
};


/**
 * Surface of genus 2
 */

class GenusTwoSurface : public ImplicitGeometry
{
protected:
    __host__ __device__ virtual float getValue(const Eigen::Vector3f& p);
    __host__ __device__ virtual Eigen::Vector3f getGradient(const Eigen::Vector3f& p);

public:
    __host__ __device__ GenusTwoSurface(const Eigen::Vector3f& range);
    __host__ __device__ virtual const VolumeSampleData& sample(const Eigen::Vector3f& p);
};


/**
 * Surface of wineglass
 */

class WineGlassSurface : public ImplicitGeometry
{
protected:
    __host__ __device__ virtual float getValue(const Eigen::Vector3f& p);
    __host__ __device__ virtual Eigen::Vector3f getGradient(const Eigen::Vector3f& p);

public:
    __host__ __device__ WineGlassSurface(const Eigen::Vector3f& range);
    __host__ __device__ virtual const VolumeSampleData& sample(const Eigen::Vector3f& p);
};


/**
 * Porous surface
 */

class PorousSurface : public ImplicitGeometry
{
protected:
    __host__ __device__ virtual float getValue(const Eigen::Vector3f& p);
    __host__ __device__ virtual Eigen::Vector3f getGradient(const Eigen::Vector3f& p);

public:
    __host__ __device__ PorousSurface(const Eigen::Vector3f& range);
    __host__ __device__ virtual const VolumeSampleData& sample(const Eigen::Vector3f& p);
};


/**
 * Discrete volume grids
 */

class Volume : public ImplicitGeometry
{
protected:
    VolumeSampleData* grid_data;
    Eigen::Vector3i grid_dimension;
    Eigen::Vector3f cell_size;

    __host__ __device__ void computeGradients();
    __host__ __device__ float centralDifference(Eigen::Vector3i coords, int axis_idx);
    __host__ __device__ Voxel getVoxel(const Eigen::Vector3f& position);
    __host__ __device__ VolumeSampleData& indexToData(const Eigen::Vector3i& index);

public:
    __host__ __device__ Volume(VolumeSampleData* grid_data, const Eigen::Vector3i& grid_dim, const Eigen::Vector3f& range);
    __host__ __device__ virtual const VolumeSampleData& sample(const Eigen::Vector3f&);
    __host__ static bool readFromFile(const char* file_path, VolumeSampleData** vol_data, Eigen::Vector3i& grid_size, Eigen::Vector3f& bbox_size);
};
