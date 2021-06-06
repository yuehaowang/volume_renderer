#include "light.hpp"


/**
 * Light class
 */

__host__ __device__ Light::Light(const Eigen::Vector3f& pos, const Eigen::Vector3f& l)
    : position(pos)
    , radiance(l)
{
}

__host__ __device__ Eigen::Vector3f Light::getPosition() const
{
    return position;
}

__host__ __device__ Eigen::Vector3f Light::getRadiance() const
{
    return radiance;
}

__host__ __device__ void Light::setRadiance(const Eigen::Vector3f& l)
{
    radiance = l;
}