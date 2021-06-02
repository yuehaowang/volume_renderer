#include "implicit_geom.hpp"
#include "constant.hpp"


/**
 * ImplicitGeometry class
 */

__host__ __device__ AABB ImplicitGeometry::getBBox() const
{
    return bounding_box;
}

__host__ __device__ bool ImplicitGeometry::bboxRayIntersection(const Ray& r, float& t_entry, float& t_exit) const
{
    return bounding_box.rayIntersection(r, t_entry, t_exit);
}


/**
 * GenusTwoSurface class
 */

__host__ __device__ GenusTwoSurface::GenusTwoSurface(Eigen::Vector3f pos, Eigen::Vector3f range)
{
    bounding_box = AABB(pos - range / 2.0, pos + range / 2.0);
}

__host__ __device__ float GenusTwoSurface::getValue(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    return 2 * y * (y * y - 3 * x * x) * (1 - z * z) + pow(x * x + y * y, 2) - (9 * z * z - 1) * (1 - z * z);
}

__host__ __device__ Eigen::Vector3f GenusTwoSurface::computeGradient(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    float dx = -12 * (1 - z * z) * y * x + 4 * pow(x, 3) + 4 * y * y * x;
    float dy = 6 * (1 - z * z) * (y * y - x * x) + 4 * pow(y, 3) + 4 * x * x * y;
    float dz = -4 * (y * y - 3 * x * x) * y * z - (20 * z - 36 * pow(z, 3));
    return Eigen::Vector3f(dx, dy, dz);
}


/**
 * WineGlassSurface class
 */

__host__ __device__ WineGlassSurface::WineGlassSurface(Eigen::Vector3f pos, Eigen::Vector3f range)
{
    bounding_box = AABB(pos - range / 2.0, pos + range / 2.0);
}

__host__ __device__ float WineGlassSurface::getValue(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    /* Out of domain: Not-A-Number */
    if (z + 3.2 <= 0)
    {
        return INFINITY;
    }

    return x * x + y * y - pow(log(z + 3.2), 2) - 0.09;
}

__host__ __device__ Eigen::Vector3f WineGlassSurface::computeGradient(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    /* Out of domain: Not-A-Number */
    if (z + 3.2 <= 0)
    {
        return Eigen::Vector3f(0, 0, 0);
    }

    float dx = 2 * x;
    float dy = 2 * y;
    float dz = -2 * log(z + 3.2) / (z + 3.2);
    return Eigen::Vector3f(dx, dy, dz);
}


/**
 * PorousSurface class
 */

__host__ __device__ PorousSurface::PorousSurface(Eigen::Vector3f pos, Eigen::Vector3f range)
{
    bounding_box = AABB(pos - range / 2.0, pos + range / 2.0);
}

__host__ __device__ float PorousSurface::getValue(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    return -0.02 + pow(-0.88 + pow(y, 2), 2) * pow(2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2), 2) + pow(-0.88 + pow(z, 2), 2) * pow(2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2), 2) + pow(-0.88 + pow(x, 2), 2) * pow(1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z), 2);
}

__host__ __device__ Eigen::Vector3f PorousSurface::computeGradient(Eigen::Vector3f p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    float dx = 68.2112 * (0. - 0.5 * x + 1. * pow(x, 3)) * (0. - 1. * pow(x, 2) + 1. * pow(x, 4) + 0.5822 * pow(y, 2)) * pow(-0.88 + pow(y, 2), 2) + 6.8 * x * pow(-0.88 + pow(x, 2), 2) * (1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z)) + 4 * x * (-0.88 + pow(x, 2)) * pow(1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z), 2);
    float dy = 6.8 * y * pow(-0.88 + pow(y, 2), 2) * (2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2)) + 4 * y * (-0.88 + pow(y, 2)) * pow(2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2), 2) + 68.2112 * (0. - 0.5 * y + 1. * pow(y, 3)) * (0. - 1. * pow(y, 2) + 1. * pow(y, 4) + 0.5822 * pow(z, 2)) * pow(-0.88 + pow(z, 2), 2);
    float dz = 6.8 * z * pow(-0.88 + pow(z, 2), 2) * (2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2)) + 4 * z * (-0.88 + pow(z, 2)) * pow(2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2), 2) + 39.712 * pow(-0.88 + pow(x, 2), 2) * (0. - 0.5 * z + 1. * pow(z, 3)) * (0. + 1. * pow(x, 2) - 1.7176 * pow(z, 2) + 1.7176 * pow(z, 4));
    return Eigen::Vector3f(dx, dy, dz);
}
