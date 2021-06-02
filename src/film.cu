#include "film.hpp"
#include "utils.hpp"


/**
 * Film class
 */

__host__ __device__ Film::Film()
    : resolution(0, 0)
{
}

__host__ __device__ Film::Film(const Eigen::Vector2i& res)
    : resolution(res.x(), res.y())
{
}

__host__ __device__ float Film::getAspectRatio() const
{
    return (float)(resolution.x()) / (float)(resolution.y());
}

