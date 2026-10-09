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

    __global__ void AddBroadcastKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ y,
        BroadcastShape shape)
    {
        int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // linear OUTPUT index
        if (idx >= shape.n) {
            return;
        }

        int64_t rem = idx;
        int64_t ia = 0;
        int64_t ib = 0;
        for (int d = shape.rank - 1; d >= 0; --d) {
            const int64_t coord = rem % shape.outDims[d];
            rem /= shape.outDims[d];
            ia += coord * shape.aStrides[d];
            ib += coord * shape.bStrides[d];
        }
        y[idx] = a[ia] + b[ib];
    }
    void AddBroadcast(const float* a, const float* b, float* y, const BroadcastShape& shape, cudaStream_t stream) {
        if (shape.n == 0) {
            return;
        }
        constexpr int kThreads = 256;
        int blocks = int((shape.n + kThreads - 1) / kThreads);
        AddBroadcastKernel << <blocks, kThreads, 0, stream >> > (a, b, y, shape);
        CUDA_CHECK(cudaGetLastError());
    }
}