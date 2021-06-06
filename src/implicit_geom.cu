#include <fstream>
#include <cmath>
#include "implicit_geom.hpp"
#include "config.hpp"
#include "interpolator.hpp"
#include "utils.hpp"


/**
 * ImplicitGeometry class
 */

__host__ __device__ ImplicitGeometry::ImplicitGeometry(const Eigen::Vector3f& range)
{
    bounding_box = AABB(-range / 2.0f, range / 2.0f);
}

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

__host__ __device__ GenusTwoSurface::GenusTwoSurface(const Eigen::Vector3f& range)
    : ImplicitGeometry(range)
{
}

__host__ __device__ float GenusTwoSurface::getValue(const Eigen::Vector3f& p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    return 2 * y * (y * y - 3 * x * x) * (1 - z * z) + pow(x * x + y * y, 2) - (9 * z * z - 1) * (1 - z * z);
}

__host__ __device__ Eigen::Vector3f GenusTwoSurface::getGradient(const Eigen::Vector3f& p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    float dx = -12 * (1 - z * z) * y * x + 4 * pow(x, 3) + 4 * y * y * x;
    float dy = 6 * (1 - z * z) * (y * y - x * x) + 4 * pow(y, 3) + 4 * x * x * y;
    float dz = -4 * (y * y - 3 * x * x) * y * z - (20 * z - 36 * pow(z, 3));
    return Eigen::Vector3f(dx, dy, dz);
}

__host__ __device__ const VolumeSampleData& GenusTwoSurface::sample(const Eigen::Vector3f& p)
{
    /* Get the point's relative position to the center of bounding box */
    Eigen::Vector3f p_ = p - bounding_box.getCenter();
    return VolumeSampleData(p_, getValue(p_), getGradient(p_));
}


/**
 * WineGlassSurface class
 */

__host__ __device__ WineGlassSurface::WineGlassSurface(const Eigen::Vector3f& range)
    : ImplicitGeometry(range)
{
}

__host__ __device__ float WineGlassSurface::getValue(const Eigen::Vector3f& p)
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

__host__ __device__ Eigen::Vector3f WineGlassSurface::getGradient(const Eigen::Vector3f& p)
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

__host__ __device__ const VolumeSampleData& WineGlassSurface::sample(const Eigen::Vector3f& p)
{
    /* Get the point's relative position to the center of bounding box */
    Eigen::Vector3f p_ = p - bounding_box.getCenter();
    return VolumeSampleData(p_, getValue(p_), getGradient(p_));
}


/**
 * PorousSurface class
 */

__host__ __device__ PorousSurface::PorousSurface(const Eigen::Vector3f& range)
    : ImplicitGeometry(range)
{
}

__host__ __device__ float PorousSurface::getValue(const Eigen::Vector3f& p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    return -0.02 + pow(-0.88 + pow(y, 2), 2) * pow(2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2), 2) + pow(-0.88 + pow(z, 2), 2) * pow(2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2), 2) + pow(-0.88 + pow(x, 2), 2) * pow(1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z), 2);
}

__host__ __device__ Eigen::Vector3f PorousSurface::getGradient(const Eigen::Vector3f& p)
{
    float x = p.x();
    float y = p.y();
    float z = p.z();

    float dx = 68.2112 * (0. - 0.5 * x + 1. * pow(x, 3)) * (0. - 1. * pow(x, 2) + 1. * pow(x, 4) + 0.5822 * pow(y, 2)) * pow(-0.88 + pow(y, 2), 2) + 6.8 * x * pow(-0.88 + pow(x, 2), 2) * (1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z)) + 4 * x * (-0.88 + pow(x, 2)) * pow(1.7 * pow(x, 2) + 2.92 * (-1 + z) * pow(z, 2) * (1 + z), 2);
    float dy = 6.8 * y * pow(-0.88 + pow(y, 2), 2) * (2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2)) + 4 * y * (-0.88 + pow(y, 2)) * pow(2.92 * (-1 + x) * pow(x, 2) * (1 + x) + 1.7 * pow(y, 2), 2) + 68.2112 * (0. - 0.5 * y + 1. * pow(y, 3)) * (0. - 1. * pow(y, 2) + 1. * pow(y, 4) + 0.5822 * pow(z, 2)) * pow(-0.88 + pow(z, 2), 2);
    float dz = 6.8 * z * pow(-0.88 + pow(z, 2), 2) * (2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2)) + 4 * z * (-0.88 + pow(z, 2)) * pow(2.92 * (-1 + y) * pow(y, 2) * (1 + y) + 1.7 * pow(z, 2), 2) + 39.712 * pow(-0.88 + pow(x, 2), 2) * (0. - 0.5 * z + 1. * pow(z, 3)) * (0. + 1. * pow(x, 2) - 1.7176 * pow(z, 2) + 1.7176 * pow(z, 4));
    return Eigen::Vector3f(dx, dy, dz);
}

__host__ __device__ const VolumeSampleData& PorousSurface::sample(const Eigen::Vector3f& p)
{
    /* Get the point's relative position to the center of bounding box */
    Eigen::Vector3f p_ = p - bounding_box.getCenter();
    return VolumeSampleData(p_, getValue(p_), getGradient(p_));
}


/**
 * Volume class
 */

__host__ __device__ Volume::Volume(VolumeSampleData* d, const Eigen::Vector3i& grid_dim, const Eigen::Vector3f& range)
    : grid_data(d)
    , grid_dimension(grid_dim)
    , cell_size(range.x() / (grid_dim.x() - 1), range.y() / (grid_dim.y() - 1), range.z() / (grid_dim.z() - 1))
    , ImplicitGeometry(range)
{
    computeGradients();
}

__host__ __device__ Volume::~Volume()
{
    delete grid_data;
}

__host__ __device__ float Volume::centralDifference(Eigen::Vector3i coords, int axis_idx)
{
    /* Handle boudary conditions. Set boundary's graident as its neighbours' gradient */
    bool on_lower_bound = coords(axis_idx) <= 0, on_upper_bound = coords(axis_idx) >= grid_dimension(axis_idx) - 1;
    if (on_lower_bound && on_upper_bound)
    {
        return 0.0f;
    }
    if (on_lower_bound)
    {
        coords(axis_idx) += 1;
    }
    else if (on_upper_bound)
    {
        coords(axis_idx) -= 1;
    }

    Eigen::Vector3i prev_coords = coords;
    prev_coords(axis_idx) -= 1;
    Eigen::Vector3i next_coords = coords;
    next_coords(axis_idx) += 1;

    const VolumeSampleData& curr = indexToData(coords);
    const VolumeSampleData& prev = indexToData(prev_coords);
    const VolumeSampleData& next = indexToData(next_coords);

    /* Find gradient */
    return (next.value - prev.value) / (cell_size[axis_idx] * 2);
}

__host__ __device__ void Volume::computeGradients()
{
    int count = grid_dimension.x() * grid_dimension.y() * grid_dimension.z();

    for (int i = 0; i < count; ++i)
    {
        int z_idx = i / (grid_dimension.y() * grid_dimension.x());
        int y_idx = (i - z_idx * grid_dimension.x() * grid_dimension.y()) / grid_dimension.x();
        int x_idx = i - z_idx * grid_dimension.x() * grid_dimension.y() - y_idx * grid_dimension.x();
        Eigen::Vector3i idx_coords(x_idx, y_idx, z_idx);

        float x_grad = centralDifference(idx_coords, 0);
        float y_grad = centralDifference(idx_coords, 1);
        float z_grad = centralDifference(idx_coords, 2);
        Eigen::Vector3f grad = Eigen::Vector3f(x_grad, y_grad, z_grad);

        grid_data[i].gradient = grad;
    }
}

__host__ __device__ Voxel Volume::getVoxel(const Eigen::Vector3f& position)
{
    Eigen::Vector3f index = (position - bounding_box.lb);
    index = index.cwiseQuotient(cell_size);
    index = index.cwiseMax(Eigen::Vector3f::Zero());
    index = index.cwiseMin((grid_dimension - Eigen::Vector3i(2, 2, 2)).cast<float>());

    Eigen::Vector3i lb(floor(index.x()), floor(index.y()), floor(index.z()));
    VolumeSampleData& c100 = indexToData(lb + Eigen::Vector3i(1, 0, 0));
    VolumeSampleData& c000 = indexToData(lb);
    VolumeSampleData& c010 = indexToData(lb + Eigen::Vector3i(0, 1, 0));
    VolumeSampleData& c110 = indexToData(lb + Eigen::Vector3i(1, 1, 0));
    VolumeSampleData& c001 = indexToData(lb + Eigen::Vector3i(0, 0, 1));
    VolumeSampleData& c101 = indexToData(lb + Eigen::Vector3i(1, 0, 1));
    VolumeSampleData& c011 = indexToData(lb + Eigen::Vector3i(0, 1, 1));
    VolumeSampleData& c111 = indexToData(lb + Eigen::Vector3i(1, 1, 1));
    return Voxel(c000, c100, c010, c110, c001, c101, c011, c111);
}

__host__ __device__ VolumeSampleData& Volume::indexToData(const Eigen::Vector3i& index)
{
    return grid_data[index.z() * grid_dimension.y() * grid_dimension.x() + index.y() * grid_dimension.x() + index.x()];
}

__host__ __device__ const VolumeSampleData& Volume::sample(const Eigen::Vector3f& p)
{
    Voxel vo = getVoxel(p);
    return Interpolator::trilinearInterpolate(p - bounding_box.lb, vo);
}

__host__ bool Volume::readFromFile(const char* file_path, VolumeSampleData** vol_data, Eigen::Vector3i& grid_dim, Eigen::Vector3f& bbox_size)
{
    printf("-- Loading %s\n", file_path);

    FILE *fp = fopen(file_path, "rb");
    if (fp)
    {
        fread(grid_dim.data(), sizeof(int), 3, fp);
        fread(bbox_size.data(), sizeof(float), 3, fp);
        int count = grid_dim.x() * grid_dim.y() * grid_dim.z();
        Eigen::Vector3f cell_size(
            bbox_size.x() / (float)(grid_dim.x() - 1),
            bbox_size.y() / (float)(grid_dim.y() - 1),
            bbox_size.z() / (float)(grid_dim.z() - 1));

        checkCudaErrors(cudaMallocManaged((void**)vol_data, count * sizeof(VolumeSampleData)));

        float val = 0;
        float min_val = VOLUME_MAX_DENSITY;
        float max_val = VOLUME_MIN_DENSITY;
        for (int curind = 0; curind < count; curind++)
        {
            int z_idx = curind / (grid_dim.y() * grid_dim.x());
            int y_idx = (curind - z_idx * grid_dim.x() * grid_dim.y()) / grid_dim.x();
            int x_idx = curind - z_idx * grid_dim.x() * grid_dim.y() - y_idx * grid_dim.x();
            fread(&val, sizeof(float), 1, fp);

            (*vol_data)[curind].position = Eigen::Vector3f(x_idx * cell_size.x(), y_idx * cell_size.y(), z_idx * cell_size.z());
            (*vol_data)[curind].value = MathUtils::clamp(val, VOLUME_MIN_DENSITY, VOLUME_MAX_DENSITY);

            min_val = min(min_val, val);
            max_val = max(max_val, val);
        }

        fclose(fp);

        printf("---- Successfully load %s\n---- Grid=(%d, %d, %d), BBoxSize=(%f, %f, %f), ValueRange=[%f, %f]\n",
            file_path, grid_dim.x(), grid_dim.y(), grid_dim.z(),
            bbox_size.x(), bbox_size.y(), bbox_size.z(),
            min_val, max_val);

        return true;
    }
    else
    {
        printf("---- Failed to load %s\n", file_path);

        return false;
    }
}
