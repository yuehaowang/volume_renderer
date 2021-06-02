#include "camera.hpp"
#include "constant.hpp"


/**
 * Camera class
 */

__host__ __device__ Camera::Camera(const Eigen::Vector3f& pos, float v_fov, const Eigen::Vector2i& film_res)
    : film(film_res)
    , vertical_fov(v_fov)
    , position(pos)
    , forward(0, 0, 1)
    , right(1, 0, 0)
    , up(0, 1, 0)
{
}

__host__ __device__ void Camera::lookAt(const Eigen::Vector3f& look_at, const Eigen::Vector3f& ref_up)
{
    forward = (position - look_at).normalized();
    up = (ref_up - (ref_up.dot(forward) * forward)).normalized();
    right = up.cross(forward);
}

__host__ __device__ Ray Camera::generateRay(float dx, float dy) const
{
    float half_fov = vertical_fov / 2;
    float image_aspect_ratio = film.getAspectRatio(); 
    float im_plane_w = tan(half_fov * PI / 180) * image_aspect_ratio;
    float im_plane_h = tan(half_fov * PI / 180);
    
    float x_cam = (2 * (dx / film.resolution.x()) - 1) * im_plane_w;
    float y_cam = (2 * (dy / film.resolution.y()) - 1) * im_plane_h;
    Eigen::Vector3f dir_camera(x_cam, y_cam, -1.0);

    Eigen::Matrix3f view;
    view.col(0) << right;
    view.col(1) << up;
    view.col(2) << forward;
    Eigen::Vector3f dir_world = (view * dir_camera).normalized();

    return Ray(position, dir_world);
}

__host__ __device__ const Film& Camera::getFilm() const
{
    return film;
}

__host__ __device__ Eigen::Vector3f Camera::getPosition() const
{
    return position;
}
