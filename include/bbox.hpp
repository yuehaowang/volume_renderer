#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "ray.hpp"


/**
 * Axis-Aligned Bounding Box
 */

struct AABB {
    Eigen::Vector3f lb;
    Eigen::Vector3f ub;

    __host__ __device__ AABB();
    /* Construct AABB by coordinates of lower bound and upper bound */
    __host__ __device__ AABB(float lb_x, float lb_y, float lb_z, float ub_x, float ub_y, float ub_z);
    __host__ __device__ AABB(Eigen::Vector3f lb, Eigen::Vector3f ub);
    /* Construct AABB for a sphere */
    __host__ __device__ AABB(const Eigen::Vector3f& pos, float radius);
    /* Construct AABB for a triangle */
    __host__ __device__ AABB(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& v3);
    /* Construct AABB by merging two AABBs */
    __host__ __device__ AABB(const AABB& a, const AABB& b);
    /* Get the AABB center */
    __host__ __device__ Eigen::Vector3f getCenter() const;
    /* Get the AABB size */
    __host__ __device__ Eigen::Vector3f getSize() const;
    /* Get the length of a specified side on the AABB */
    __host__ __device__ float getDist(int c) const;
    /* Get the volume of the AABB */
    __host__ __device__ float getVolume() const;
    /* Check whether the AABB is overlapping with another AABB */
    __host__ __device__ bool isOverlap(const AABB& a) const;
    /* Get the diagonal length */
    __host__ __device__ float diagonalLength() const;
    /* Test intersection with a ray */
    __host__ __device__ bool rayIntersection(const Ray& ray, float& t_in, float& t_out) const;
};

