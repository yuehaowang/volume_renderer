#include "classifier.hpp"
#include "constant.hpp"
#include "config.hpp"
#include "utils.hpp"


/**
 * Classifier class
 */

__host__ __device__ Classifier::Classifier(float K, VisualizationTarget vis_tgt, tinycolormap::ColormapType cm)
    : kappa(K)
    , cm_type(cm)
    , visualize_target(vis_tgt)
{
}

__host__ __device__ Eigen::Vector3f Classifier::getMappedColor(float value, const Eigen::Vector3f& normal) const
{
    switch(visualize_target)
    {
        case VisualizationTarget::NORMAL_X:
            return tinycolormap::GetColor(MathUtils::clamp((normal.x() + 1.0) / 2.0, 0.0f, 1.0f), cm_type).ConvertToEigen().cast<float>();

        case VisualizationTarget::NORMAL_Y:
            return tinycolormap::GetColor(MathUtils::clamp((normal.y() + 1.0) / 2.0, 0.0f, 1.0f), cm_type).ConvertToEigen().cast<float>();

        case VisualizationTarget::NORMAL_Z:
            return tinycolormap::GetColor(MathUtils::clamp((normal.z() + 1.0) / 2.0, 0.0f, 1.0f), cm_type).ConvertToEigen().cast<float>();

        case VisualizationTarget::VALUE:
            return tinycolormap::GetColor(MathUtils::clamp(value, 0.0f, 1.0f), cm_type).ConvertToEigen().cast<float>();
    }
}


/**
 * IsosurfaceClassifier class
 */

__host__ __device__ IsosurfaceClassifier::IsosurfaceClassifier()
    : isovalue(0.0f)
    , sigma(0.01f)
    , Classifier(4000.0f, VisualizationTarget::NORMAL_X)
{
}

__host__ __device__ float IsosurfaceClassifier::transfer(const VolumeSampleData& v_data) const
{
    return MathUtils::Gaussian(isovalue, sigma, v_data.value);
}

__host__ __device__ OpticsData IsosurfaceClassifier::transfer(const VolumeSampleData& v_data, const Camera* cam, Light** lis, int lis_num, float dt) const
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
            
            /* Retrieve the color for visualization */
            Eigen::Vector3f surface_color = getMappedColor(val, normal);

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

    optics.transparency = Eigen::Vector3f::Ones() * exp(-kappa * val * dt);
    optics.color = 1000.0f * q * val * dt;

    return optics;
}


/**
 * VolumeClassifier class
 */

__host__ __device__ VolumeClassifier::VolumeClassifier()
    : min_transparency(0.0f)
    , Classifier(1.0f, VisualizationTarget::VALUE)
{
}

__host__ __device__ float VolumeClassifier::transfer(const VolumeSampleData& v_data) const
{
    return v_data.value;
}

__host__ __device__ OpticsData VolumeClassifier::transfer(const VolumeSampleData& v_data, const Camera* cam, Light** lis, int lis_num, float dt) const
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
            
            /* Retrieve the color for visualization */
            Eigen::Vector3f xsec_color = getMappedColor(val, normal);

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
            q += (C_a + C_d + C_s).cwiseProduct(xsec_color);
        }
    }

    optics.transparency = Eigen::Vector3f::Ones() * (exp(-val * kappa * dt) * (1.0 - min_transparency) + min_transparency);
    optics.color = q * val * dt;

    return optics;
}