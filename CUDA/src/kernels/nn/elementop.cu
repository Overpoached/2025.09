#include "nn.h"

#include "../kernel_utility.h"

#include <vector>
#include <math_constants.h>

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


    namespace {
        constexpr int kWarpSize = 32;

        __device__ float WarpMax(float v) {
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
            }
            return v;
        }

        __device__ float WarpSum(float v) {
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, offset);
            }
            return v;
        }
    }

    // One warp per row. Lanes stride across the row; reductions use register shuffles only.
    __global__ void SoftmaxKernel(const float* __restrict__ x,
        float* __restrict__ y,
        SoftmaxShape shape)
    {
        const int lane = threadIdx.x % kWarpSize;
        const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / kWarpSize;
        if (row >= shape.rows) {
            return;   // all 32 lanes share the same row, so the whole warp exits together: shuffles stay safe
        }

        // Row (o, in) of [outer, cols, inner]: element j lives at base + j * inner
        const int64_t o = row / shape.inner;
        const int64_t in = row % shape.inner;
        const int64_t base = o * shape.cols * shape.inner + in;
        const float* xr = x + base;
        float* yr = y + base;

        // Pass 1: row max (subtracting it keeps exp() from overflowing)
        float m = -CUDART_INF_F;
        for (int64_t j = lane; j < shape.cols; j += kWarpSize) {
            m = fmaxf(m, xr[j * shape.inner]);
        }
        m = WarpMax(m);

        // Pass 2: sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t j = lane; j < shape.cols; j += kWarpSize) {
            sum += expf(xr[j * shape.inner] - m);
        }
        sum = WarpSum(sum);
        const float inv = 1.0f / sum;

        // Pass 3: normalise
        for (int64_t j = lane; j < shape.cols; j += kWarpSize) {
            yr[j * shape.inner] = expf(xr[j * shape.inner] - m) * inv;
        }
    }

    void Softmax(const float* x, float* y, const SoftmaxShape& shape, cudaStream_t stream)
    {
        if (shape.rows == 0 || shape.cols == 0) {
            return;
        }
        constexpr int kWarpsPerBlock = 8;
        constexpr int kThreads = kWarpsPerBlock * kWarpSize;   // 256
        const int64_t blocks = (shape.rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
        SoftmaxKernel << <static_cast<unsigned int>(blocks), kThreads, 0, stream >> > (x, y, shape);
        CUDA_CHECK(cudaGetLastError());
    }
}