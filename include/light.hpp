#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>


/**
 * Point light
 */

class Light
{
protected:
	Eigen::Vector3f position;
	Eigen::Vector3f radiance;

public:
	__host__ __device__ Light(const Eigen::Vector3f& pos, const Eigen::Vector3f& l);
	__host__ __device__ virtual Eigen::Vector3f getPosition() const;
	__host__ __device__ virtual Eigen::Vector3f getRadiance() const;
	__host__ __device__ virtual void setRadiance(const Eigen::Vector3f& l);
};
