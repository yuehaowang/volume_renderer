#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "voxel.hpp"


/**
 * Interpolators for points inside voxels
 */

namespace Interpolator
{
    __host__ __device__ VolumeSampleData trilinearInterpolate(const Eigen::Vector3f& p, const Voxel& voxel);
};
