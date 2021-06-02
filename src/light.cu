#include "light.hpp"


/**
 * Light class
 */

__host__ __device__ Light::Light(Eigen::Vector3f pos, Eigen::Vector3f rgb)
    : position(pos)
    , color(rgb)
{
}

__host__ __device__ Eigen::Vector3f Light::getPosition() const
{
    return position;
}

__host__ __device__ Eigen::Vector3f Light::getColor() const
{
    return color;
}
