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

    // Merges adjacent dims that are contiguous for BOTH operands (or broadcast in both).
    // Equal shapes collapse to rank 1; [4,50,768] + [50,768] becomes [4, 38400] with b strides [0, 1].
    void CoalesceDims(BroadcastShape& s) {
        int64_t dims[kMaxBroadcastRank], sa[kMaxBroadcastRank], sb[kMaxBroadcastRank];
        int r = 0;   // built innermost-first
        for (int d = s.rank - 1; d >= 0; --d) {
            if (s.outDims[d] == 1) {
                continue;   // size-1 output dim: coordinate is always 0, contributes nothing
            }
            if (r > 0 && s.aStrides[d] == sa[r - 1] * dims[r - 1] && s.bStrides[d] == sb[r - 1] * dims[r - 1]) {
                dims[r - 1] *= s.outDims[d];   // contiguous with the dim inside it, for both operands: merge
            }
            else {
                dims[r] = s.outDims[d];
                sa[r] = s.aStrides[d];
                sb[r] = s.bStrides[d];
                ++r;
            }
        }
        if (r == 0) {   // every dim was 1: a single element
            dims[0] = 1; sa[0] = 0; sb[0] = 0; r = 1;
        }
        s.rank = r;
        for (int i = 0; i < r; ++i) {   // store outermost-first, as the kernel expects
            s.outDims[i] = dims[r - 1 - i];
            s.aStrides[i] = sa[r - 1 - i];
            s.bStrides[i] = sb[r - 1 - i];
        }
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
        CoalesceDims(s); // outShape (the tensor's shape) is untouched; only the kernel's view is simplified
        return s;
    }

    __global__ void AddKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ y,
        BroadcastShape shape) 
    {
        int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // linear OUTPUT index
        if (idx >= shape.n) {
            return;
        }

        int64_t ia, ib;
        if (shape.rank == 1) {
            // Same shapes (strides 1) or scalar broadcast (stride 0): no div/mod.
            // Uniform across the whole grid, so no divergence.
            ia = idx * shape.aStrides[0];
            ib = idx * shape.bStrides[0];
        }
        else {
            int64_t rem = idx;
            ia = 0;
            ib = 0;
            for (int d = shape.rank - 1; d >= 0; --d) {
                const int64_t coord = rem % shape.outDims[d];
                rem /= shape.outDims[d];
                ia += coord * shape.aStrides[d];
                ib += coord * shape.bStrides[d];
            }
        }
        y[idx] = a[ia] + b[ib];
    }
    void Add(const float* a, const float* b, float* y, const BroadcastShape& shape, cudaStream_t stream) {
        if (shape.n == 0) {
            return;
        }
        constexpr int kThreads = 256;
        int blocks = int((shape.n + kThreads - 1) / kThreads);
        AddKernel << <blocks, kThreads, 0, stream >> > (a, b, y, shape);
        CUDA_CHECK(cudaGetLastError());
    }

    // Same-shape convenience overload: a rank-1 shape with unit strides
    void Add(const float* a, const float* b, float* y, int64_t n, cudaStream_t stream) {
        BroadcastShape s{};
        s.rank = 1;
        s.n = n;
        s.outDims[0] = n;
        s.aStrides[0] = 1;
        s.bStrides[0] = 1;
        Add(a, b, y, s, stream);
    }


    __global__ void MulKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ y,
        BroadcastShape shape)
    {
        int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // linear OUTPUT index
        if (idx >= shape.n) {
            return;
        }

        int64_t ia, ib;
        if (shape.rank == 1) {
            // Same shapes (strides 1) or scalar broadcast (stride 0): no div/mod.
            // Uniform across the whole grid, so no divergence.
            ia = idx * shape.aStrides[0];
            ib = idx * shape.bStrides[0];
        }
        else {
            int64_t rem = idx;
            ia = 0;
            ib = 0;
            for (int d = shape.rank - 1; d >= 0; --d) {
                const int64_t coord = rem % shape.outDims[d];
                rem /= shape.outDims[d];
                ia += coord * shape.aStrides[d];
                ib += coord * shape.bStrides[d];
            }
        }
        y[idx] = a[ia] * b[ib];
    }
    void Mul(const float* a, const float* b, float* y, const BroadcastShape& shape, cudaStream_t stream) {
        if (shape.n == 0) {
            return;
        }
        constexpr int kThreads = 256;
        int blocks = int((shape.n + kThreads - 1) / kThreads);
        MulKernel << <blocks, kThreads, 0, stream >> > (a, b, y, shape);
        CUDA_CHECK(cudaGetLastError());
    }

    // Same-shape convenience overload: a rank-1 shape with unit strides
    void Mul(const float* a, const float* b, float* y, int64_t n, cudaStream_t stream) {
        BroadcastShape s{};
        s.rank = 1;
        s.n = n;
        s.outDims[0] = n;
        s.aStrides[0] = 1;
        s.bStrides[0] = 1;
        Mul(a, b, y, s, stream);
    }

}