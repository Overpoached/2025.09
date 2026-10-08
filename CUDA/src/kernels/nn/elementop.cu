#include "nn.h"

#include "../kernel_utility.h"

namespace kernels {
    __global__ void SigmoidKernel(const float* x, float* y, int64_t n) {
        int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // one thread per element
        if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    void Sigmoid(const float* x, float* y, int64_t n, cudaStream_t stream) {
        constexpr int kThreads = 256;
		int blocks = int((n + kThreads - 1) / kThreads); // ceil (n / kThreads)
        SigmoidKernel << <blocks, kThreads, 0, stream >> > (x, y, n);
        CUDA_CHECK(cudaGetLastError());              // catches bad launch configs immediately
    }

    __global__ void AddKernel(const float* x, const float* y, float* z, int64_t n) {
        int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // one thread per element
        if (i < n) z[i] = x[i] + y[i];
    }
    void Add(const float* x, const float* y, float* z, int64_t n, cudaStream_t stream) {
        constexpr int kThreads = 256;
        int blocks = int((n + kThreads - 1) / kThreads);
        AddKernel << <blocks, kThreads, 0, stream >> > (x, y, z, n);
        CUDA_CHECK(cudaGetLastError());              // catches bad launch configs immediately
    }
}