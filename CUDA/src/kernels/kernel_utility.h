#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR() { \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) { \
		throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
	} \
}

#define CUDA_CHECK(expr) do { \
    const cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorName(err_) + \
                                 " (" + cudaGetErrorString(err_) + ") at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)