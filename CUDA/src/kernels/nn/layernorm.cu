#include "nn.h"

#include "../kernel_utility.h"

/*
* What LayerNorm does

* CLIP turns the image into 50 tokens (1 CLS plus 49 patches), and each token is a vector of 768 numbers. As those vectors pass through the 12 transformer layers, their values can drift. Some tokens grow very large, others shrink toward zero, and every layer has to cope with a different scale. LayerNorm resets the scale before each major step.

* For each token separately (each row of 768 numbers):

* Centre it. Compute the mean of the 768 values and subtract it, so the row averages 0.
* Rescale it. Compute the variance (the average squared distance from the mean) and divide by sqrt(variance + ε), so the row has a spread of 1. ε (1e-5) prevents division by zero when a row is nearly constant.
* Apply the learned per-feature adjustment. Multiply by scale[j] (often called γ, gamma) and add bias[j] (β, beta). Both are 768-long vectors loaded from the model, and they let the network partly undo step 2 for features it wants larger or offset.

*/

namespace kernels {
    namespace {
        constexpr int kWarpSize = 32;

        // Sum across the 32 lanes of a warp using register shuffles (no shared memory).
        __device__ float WarpSum(float v) {
            // reduction here
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, offset);
            }
            return v;
        }

        // Sum across the whole block. Every thread gets the result. blockDim.x must be a multiple of 32.
        __device__ float BlockSum(float v, float* scratch /* [32] shared */) {
            const int lane = threadIdx.x % kWarpSize;
            const int warp = threadIdx.x / kWarpSize;
            const int numWarps = blockDim.x / kWarpSize;

            v = WarpSum(v);                       // 1. each warp reduces its 32 values
            if (lane == 0) {
                scratch[warp] = v;                // 2. one partial per warp into shared memory
            }
            __syncthreads();

            v = (threadIdx.x < numWarps) ? scratch[threadIdx.x] : 0.0f;
            if (warp == 0) {
                v = WarpSum(v);                   // 3. first warp reduces the partials
            }
            if (threadIdx.x == 0) {
                scratch[0] = v;
            }
            __syncthreads();

            const float total = scratch[0];       // 4. broadcast to all threads
            __syncthreads();                      // scratch may be reused by the next BlockSum
            return total;
        }
    }

    // One block per row. Threads stride across the row's columns.
    __global__ void LayerNormKernel(const float* __restrict__ x,
        const float* __restrict__ scale,
        const float* __restrict__ bias,
        float* __restrict__ y,
        int64_t rows, int64_t cols, float epsilon)
    {
        __shared__ float scratch[kWarpSize];

        const int64_t row = blockIdx.x;
        if (row >= rows) {
            return;   // uniform across the block, so no thread is left waiting at __syncthreads
        }
        const float* xr = x + row * cols;
        float* yr = y + row * cols;

        // Pass 1: mean
        float sum = 0.0f;
        for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
            sum += xr[i];
        }
        const float mean = BlockSum(sum, scratch) / static_cast<float>(cols);

        // Pass 2: variance around the mean (two-pass avoids E[x^2]-E[x]^2 cancellation)
        float sq = 0.0f;
        for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
            const float d = xr[i] - mean;
            sq += d * d;
        }
        const float var = BlockSum(sq, scratch) / static_cast<float>(cols);
        const float invStd = 1.0f / sqrtf(var + epsilon);

        // Pass 3: normalise, scale, shift
        for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
            const float n = (xr[i] - mean) * invStd;
            yr[i] = n * scale[i] + (bias ? bias[i] : 0.0f);
        }
    }

    void LayerNorm(const float* x, const float* scale, const float* bias, float* y,
        int64_t rows, int64_t cols, float epsilon, cudaStream_t stream)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        constexpr int kThreads = 256;   // multiple of 32 (BlockSum requires it); 768 cols -> 3 per thread
        LayerNormKernel << <static_cast<unsigned int>(rows), kThreads, 0, stream >> > (
            x, scale, bias, y, rows, cols, epsilon);
        CUDA_CHECK(cudaGetLastError());
    }
}