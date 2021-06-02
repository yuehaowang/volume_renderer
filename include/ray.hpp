#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>

#define RAY_MIN_T 1e-5f
#define RAY_MAX_T 1e+30f

/**
 * Data structure representing a ray
 */

struct Ray
{
    /* Original point of the ray */
    Eigen::Vector3f origin;
    /* Direction of the ray */
    Eigen::Vector3f direction;
    /* The maximum and minimum value of the parameter t */
    float range_min;
    float range_max;

    __host__ __device__ Ray()
        : origin(0, 0, 0)
        , direction(0, 0, 0)
        , range_min(RAY_MIN_T)
        , range_max(RAY_MAX_T)
    {
    }

    __host__ __device__ Ray(const Eigen::Vector3f& ori, const Eigen::Vector3f& dir, float mini_t = 1e-5f, float maxi_t = 1e+30f)
    {
        origin = ori;
        direction = dir.normalized();
        range_min = mini_t;
        range_max = maxi_t;
    }

    __host__ __device__ Eigen::Vector3f getPoint(float t) const {
        return origin + t * direction;
    }
};
