#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>


/**
 * Unreachable statement
 */

#define UNREACHABLE std::cout << "Error: Unreachable code executed. Exit(-1)..." << std::endl; exit(-1);


/**
 * String processing utilities
 */

namespace StrUtils
{
    /* Trim from left */
    __host__ void ltrim(std::string& s);
    /* Trim from right */
    __host__ void rtrim(std::string& s);
    /* Trim from both left and right */
    __host__ void trim(std::string& s);
};


/**
 * Mathematical utilities
 */

namespace MathUtils
{
    /* Clamps the input x to the closed range [lo, hi] */
    __host__ __device__ float clamp(float x, float lo, float hi);
    /* Guassian function */
    __host__ __device__ float Gaussian(float mu, float sigma, float x);
};


/**
 * Utilities for checking CUDA errors
 */

#define SINGLE_THREAD if (threadIdx.x != 0 || blockIdx.x != 0) { return; }

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )

__host__ void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line);

template<typename T>
__host__ T* cudaToDevice(T* h_ptr)
{
    T* d_ptr;
    checkCudaErrors(cudaMallocManaged((void**)&d_ptr, sizeof(T)));
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr, sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    return d_ptr;
}


/**
 * Image processing utilities
 */

namespace ImageUtils
{
    /* Performs Gamma correction on x and outputs an integer between 0-255. */
    __host__ __device__ unsigned char gammaCorrection(float x);
    /* Converts RGB pixel array to byte array with 4 channels (RGBA) */
    __global__ void pixelArrayToBytesRGBA(Eigen::Vector3f* pix_arr, unsigned char* bytes, int res_x, int res_y);
};
