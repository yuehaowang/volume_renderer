#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include "utils.hpp"


/**
 * A data structure for storing imaging pixels
 */

struct Film
{
    Eigen::Vector2i resolution;

    __host__ __device__ Film();
    __host__ __device__ Film(const Eigen::Vector2i& res);
	__host__ __device__ float getAspectRatio() const;
};
