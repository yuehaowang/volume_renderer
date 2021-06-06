#pragma once
#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime.h>


/**
 * A data structure containing information of each point in the volume
 */

struct VolumeSampleData
{
    float value;
    Eigen::Vector3f gradient;
    Eigen::Vector3f position;

    __host__ __device__ VolumeSampleData();
    __host__ __device__ VolumeSampleData(Eigen::Vector3f p, float val, Eigen::Vector3f grad);
};


/**
 * A data structure representing voxels with 8 vertices
 */

struct Voxel
{
    VolumeSampleData& c000;
    VolumeSampleData& c100;
    VolumeSampleData& c010;
    VolumeSampleData& c110;
    VolumeSampleData& c001;
    VolumeSampleData& c101;
    VolumeSampleData& c011;
    VolumeSampleData& c111;

    __host__ __device__ Voxel(
        VolumeSampleData& c000, VolumeSampleData& c100, VolumeSampleData& c010, VolumeSampleData& c110,
        VolumeSampleData& c001, VolumeSampleData& c101, VolumeSampleData& c011, VolumeSampleData& c111);
};

