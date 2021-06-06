#include "voxel.hpp"


/**
 * VolumeSampleData struct
 */

__host__ __device__ VolumeSampleData::VolumeSampleData()
    : position(Eigen::Vector3f(0, 0, 0))
    , value(nanf(""))
    , gradient(Eigen::Vector3f(0, 0, 0))
{
}

__host__ __device__ VolumeSampleData::VolumeSampleData(Eigen::Vector3f p, float val, Eigen::Vector3f grad)
    : position(p)
    , value(val)
    , gradient(grad)
{
}


/**
 * Voxel struct
 */

__host__ __device__ Voxel::Voxel(
    VolumeSampleData& c000, VolumeSampleData& c100, VolumeSampleData& c010, VolumeSampleData& c110,
    VolumeSampleData& c001, VolumeSampleData& c101, VolumeSampleData& c011, VolumeSampleData& c111)
    : c000(c000), c100(c100), c010(c010), c110(c110), c001(c001), c101(c101), c011(c011), c111(c111)
{
}