#include "nn.h"

#include "../kernel_utility.h"

#include <vector>

namespace kernels {
    namespace {
        template <typename T>
        std::string ToString(const std::vector<T>& vec, uint32_t pad = 0) {
            std::string ret = "[";
            for (auto& v : vec) {
                if (ret.size() > 1) { ret += ", "; }
                ret += std::to_string(v);
            }
            return ret + "]";
        }
    }
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
    // Numpy broadcast of a and b. Fills strides with 0 on broadcast axes. Throws on incompatible shapes.
    kernels::BroadcastShape MakeBroadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
        std::vector<int64_t>& outShape)
    {
        const int64_t rank = static_cast<int64_t>(std::max(a.size(), b.size()));
        if (rank > kernels::kMaxBroadcastRank) {
            throw std::runtime_error("Broadcast: rank too large");
        }

        // Right-align: pad the shorter shape with leading 1s
        std::vector<int64_t> pa(rank - a.size(), 1), pb(rank - b.size(), 1);
        pa.insert(pa.end(), a.begin(), a.end());
        pb.insert(pb.end(), b.begin(), b.end());

        kernels::BroadcastShape s{};
        s.rank = static_cast<int32_t>(rank);
        outShape.assign(rank, 1);
        int64_t strideA = 1, strideB = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (pa[d] != pb[d] && pa[d] != 1 && pb[d] != 1) {
                throw std::runtime_error("Broadcast: incompatible shapes " + ToString(a)
                    + " and " + ToString(b));
            }
            outShape[d] = std::max(pa[d], pb[d]);
            s.outDims[d] = outShape[d];
            s.aStrides[d] = (pa[d] == 1) ? 0 : strideA;   // size-1 axis: repeat the same element
            s.bStrides[d] = (pb[d] == 1) ? 0 : strideB;
            strideA *= pa[d];
            strideB *= pb[d];
        }
        s.n = 1;
        for (int64_t v : outShape) s.n *= v;
        return s;
    }

}