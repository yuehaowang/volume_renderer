#include "interpolator.hpp"


/* Convert a float-point number to an 1-D vector */
#define float2vec(V) ((V) * (Eigen::VectorXf::Ones(1)))


/**
 * Interpolator namespace
 */

__host__ __device__ VolumeSampleData Interpolator::trilinearInterpolate(const Eigen::Vector3f& p, const Voxel& voxel)
{
    const VolumeSampleData& c100_d = voxel.c100;
    const VolumeSampleData& c010_d = voxel.c010;
    const VolumeSampleData& c000_d = voxel.c000;
    const VolumeSampleData& c110_d = voxel.c110;
    const VolumeSampleData& c001_d = voxel.c001;
    const VolumeSampleData& c101_d = voxel.c101;
    const VolumeSampleData& c011_d = voxel.c011;
    const VolumeSampleData& c111_d = voxel.c111;

    /* Trilinear interpolation. Note: order of arguments matters.
        Reference: https://en.wikipedia.org/wiki/Trilinear_interpolation */
    float x = p.x(), y = p.y(), z = p.z();
    float x0 = c000_d.position.x(), y0 = c000_d.position.y(), z0 = c000_d.position.z();
    float x1 = c111_d.position.x(), y1 = c111_d.position.y(), z1 = c111_d.position.z();
    float xd = (x - x0) / (x1 - x0), yd = (y - y0) / (y1 - y0), zd = (z - z0) / (z1 - z0);

    /* Interpolate value */
    float val;
    {
        float c00 = c000_d.value * (1 - xd) + c100_d.value * xd;
        float c01 = c001_d.value * (1 - xd) + c101_d.value * xd;
        float c10 = c010_d.value * (1 - xd) + c110_d.value * xd;
        float c11 = c011_d.value * (1 - xd) + c111_d.value * xd;
        float c0 = c00 * (1 - yd) + c10 * yd;
        float c1 = c01 * (1 - yd) + c11 * yd;
        val = c0 * (1 - zd) + c1 * zd;
    }
    
    /* Interpolate gradient */
    Eigen::Vector3f grad;
    {
        Eigen::Vector3f c00 = c000_d.gradient * (1 - xd) + c100_d.gradient * xd;
        Eigen::Vector3f c01 = c001_d.gradient * (1 - xd) + c101_d.gradient * xd;
        Eigen::Vector3f c10 = c010_d.gradient * (1 - xd) + c110_d.gradient * xd;
        Eigen::Vector3f c11 = c011_d.gradient * (1 - xd) + c111_d.gradient * xd;
        Eigen::Vector3f c0 = c00 * (1 - yd) + c10 * yd;
        Eigen::Vector3f c1 = c01 * (1 - yd) + c11 * yd;
        grad = c0 * (1 - zd) + c1 * zd;
    }

    return VolumeSampleData(p, val, grad);
}
