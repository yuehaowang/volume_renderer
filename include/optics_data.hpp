#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>


/**
 * A data structure for storing optical data
 */

struct OpticsData
{
    Eigen::Vector3f transparency = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    Eigen::Vector3f color = Eigen::Vector3f(0, 0, 0);
    __host__ __device__ Eigen::Vector3f getColor() const
    {
        return color;
    }
    __host__ __device__ Eigen::Vector3f getOpacity()
    {
        return Eigen::Vector3f::Ones() - this->transparency;
    }
};
