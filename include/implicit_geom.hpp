#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "bbox.hpp"
#include "ray.hpp"


/**
 * A data structure containing information of each point in the volume
 */

struct VolumeSampleData
{
    float value;
    Eigen::Vector3f gradient;
    Eigen::Vector3f position;
};


/**
 * Base class of implicit geometries
 */

class ImplicitGeometry
{
protected:
    AABB bounding_box;

public:
    __host__ __device__ AABB getBBox() const;
    __host__ __device__ virtual float getValue(Eigen::Vector3f p) = 0;
    __host__ __device__ virtual Eigen::Vector3f computeGradient(Eigen::Vector3f p) = 0;
    __host__ __device__ virtual bool bboxRayIntersection(const Ray& r, float& t_entry, float& t_exit) const;
};


/**
 * Surface of genus 2
 */

class GenusTwoSurface : public ImplicitGeometry
{
public:
    __host__ __device__ GenusTwoSurface(Eigen::Vector3f pos, Eigen::Vector3f range);
    __host__ __device__ virtual float getValue(Eigen::Vector3f p);
    __host__ __device__ virtual Eigen::Vector3f computeGradient(Eigen::Vector3f p);
};


/**
 * Surface of wineglass
 */

class WineGlassSurface : public ImplicitGeometry
{
public:
    __host__ __device__ WineGlassSurface(Eigen::Vector3f pos, Eigen::Vector3f range);
    __host__ __device__ virtual float getValue(Eigen::Vector3f p);
    __host__ __device__ virtual Eigen::Vector3f computeGradient(Eigen::Vector3f p);
};


/**
 * Porous surface
 */

class PorousSurface : public ImplicitGeometry
{
public:
    __host__ __device__ PorousSurface(Eigen::Vector3f pos, Eigen::Vector3f range);
    __host__ __device__ virtual float getValue(Eigen::Vector3f p);
    __host__ __device__ virtual Eigen::Vector3f computeGradient(Eigen::Vector3f p);
};
