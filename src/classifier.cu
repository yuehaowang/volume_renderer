#define TINYCOLORMAP_WITH_EIGEN
#include "tinycolormap_cuda.hpp"
#include "classifier.hpp"
#include "constant.hpp"
#include "config.hpp"
#include "utils.hpp"


/**
 * IsosurfaceClassifier class
 */

__host__ __device__ IsosurfaceClassifier::IsosurfaceClassifier(float isoval)
    : isovalue(isoval)
{
}

__host__ __device__ float IsosurfaceClassifier::transfer(VolumeSampleData v_data) const
{
    return MathUtils::Gaussian(isovalue, 0.01, v_data.value);
}

__host__ __device__ OpticsData IsosurfaceClassifier::transfer(VolumeSampleData v_data, const Camera* cam, Light** lis, int lis_num, float dt) const
{
    OpticsData optics;

    float val = transfer(v_data);

    /* Emission */
    Eigen::Vector3f q(0, 0, 0);

    if (val > DELTA)
    {
        for (int li_idx = 0; li_idx < lis_num; ++li_idx)
        {
            const Light* li = lis[li_idx];
            /* Compute view-in direction */
            Eigen::Vector3f view_dir = (cam->getPosition() - v_data.position).normalized();
            /* Compute normal */
            Eigen::Vector3f normal = v_data.gradient.normalized();
            
            /* Visualize the x-component of normals */
            Eigen::Vector3f surface_color = tinycolormap::GetColor((normal.x() + 1.0) / 2.0, tinycolormap::ColormapType::Turbo).ConvertToEigen().cast<float>();

            /* Ambient */
            Eigen::Vector3f C_a = li->getRadiance() * AMBIENT_MAGNITUDE;

            /* Compute light direction, light path distance, etc. */
            Eigen::Vector3f pos_diff = li->getPosition() - v_data.position;
            float li_dist = pos_diff.norm();
            Eigen::Vector3f li_dir = pos_diff / li_dist;

            /* Diffusion */
            float diff = max(normal.dot(li_dir), 0.0f);
            Eigen::Vector3f C_d = diff * li->getRadiance();

            /* Specular */
            Eigen::Vector3f halfway_dir = (view_dir + li_dir).normalized();
            float spec = pow(max(normal.dot(halfway_dir), 0.0f), SPECULAR_SHININESS);
            Eigen::Vector3f C_s = spec * li->getRadiance();

            /* Add the emission contribution */
            q += (C_a + C_d + C_s).cwiseProduct(surface_color);
        }
    }

    optics.transparency = Eigen::Vector3f::Ones() * exp(-4000 * val * dt);
    optics.color = q * val * dt;

    return optics;
}

