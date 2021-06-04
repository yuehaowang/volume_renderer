#include <random>
#include <algorithm>
#include "utils.hpp"
#include "constant.hpp"


/**
 * StrUtils namespace
 */

void StrUtils::ltrim(std::string& s)
{
   s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
}

void StrUtils::rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

void StrUtils::trim(std::string& s)
{
    ltrim(s);
    rtrim(s);
}


/**
 * MathUtils namespace
 */

__host__ __device__ float MathUtils::clamp(float x, float lo, float hi)
{
    return x < lo ? lo : x > hi ? hi : x;
}

__host__ __device__ float MathUtils::Gaussian(float mu, float sigma, float x)
{
    if (std::isinf(x) || std::isnan(x))
    {
        return 0;
    }
    return exp(-pow(x - mu, 2) / (2 * sigma * sigma));
}


/**
 * Function: checkCuda
 */

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        /* Make sure calling CUDA Device Reset before exiting */
        cudaDeviceReset();
        exit(99);
    }
}


/**
 * ImageUtils namespace
 */


__host__ __device__ unsigned char ImageUtils::gammaCorrection(float x)
{
	return (unsigned char)(pow(MathUtils::clamp(x, 0.0, 1.0), 1 / 2.2) * 255);
}

__global__ void ImageUtils::pixelArrayToBytesRGBA(Eigen::Vector3f* pix_arr, unsigned char* bytes, int res_x, int res_y)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if((dx >= res_x) || (dy >= res_y))
    {
        return;
    }

    int i = dy * res_x + dx;
    /* Byte arrays' +y axes are downward */
    int j = (res_y - dy) * res_x + dx;

    bytes[j * 4] = ImageUtils::gammaCorrection(pix_arr[i].x());
    bytes[j * 4 + 1] = ImageUtils::gammaCorrection(pix_arr[i].y());
    bytes[j * 4 + 2] = ImageUtils::gammaCorrection(pix_arr[i].z());
    bytes[j * 4 + 3] = 255;
}
