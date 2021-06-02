#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include "film.hpp"
#include "ray.hpp"


/**
 * Pin-hole camera model
 */

class Camera
{
protected:
    Eigen::Vector3f position;
    Eigen::Vector3f forward;
    Eigen::Vector3f right;
    Eigen::Vector3f up;
    float vertical_fov;
    Film film;

public:
    __host__ __device__ Camera(const Eigen::Vector3f& pos, float v_fov, const Eigen::Vector2i& film_res);
	__host__ __device__ void lookAt(const Eigen::Vector3f& look_at, const Eigen::Vector3f& ref_up);
    __host__ __device__ Ray generateRay(float dx, float dy) const;
    __host__ __device__ const Film& getFilm() const;
    __host__ __device__ Eigen::Vector3f getPosition() const;
    __host__ __device__ void moveTo(Eigen::Vector3f des);
};
