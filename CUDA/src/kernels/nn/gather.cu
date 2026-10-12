#include "nn.h"

#include "../kernel_utility.h"

/*
The following few examples illustrate how Gather works for specific shapes of data, indices, and given value of axis:

data shape  indices shape   axis    output shape    output equation
(P, Q)      ( ) (a scalar)  0       (Q)             output[q] = data[indices, q]
(P, Q, R)   ( ) (a scalar)  1       (P, R)          output[p, r] = data[p, indices, r]
(P, Q)      (R, S)          0       (R, S, Q)       output[r, s, q] = data[ [indices[r, s], q]
(P, Q)      (R, S)          1       (P, R, S)       output[p, r, s] = data[ p, indices[r, s]]
*/

namespace kernels {
    __global__ void GatherKernel(const float* __restrict__ data,
        const int64_t* __restrict__ indices,
        float* __restrict__ y,
        GatherShape shape)
    {
        const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // linear OUTPUT index
        if (i >= shape.n) {
            return;
        }

        // Output is [outer, numIndices, inner]: peel off the coordinates
        const int64_t in = i % shape.inner;
        const int64_t t = i / shape.inner;
        const int64_t j = t % shape.numIndices;
        const int64_t o = t / shape.numIndices;

        int64_t idx = indices[j];
        if (idx < 0) {
            idx += shape.axisDim;   // ONNX: negative counts from the end
        }
        if (idx < 0 || idx >= shape.axisDim) {
            y[i] = __int_as_float(0x7fc00000);   // out of range: poison with NaN instead of reading out of bounds
            return;
        }

        y[i] = data[(o * shape.axisDim + idx) * shape.inner + in];
    }

    void Gather(const float* data, const int64_t* indices, float* y, const GatherShape& shape, cudaStream_t stream)
    {
        if (shape.n == 0) {
            return;
        }
        constexpr int kThreads = 256;
        const int64_t blocks = (shape.n + kThreads - 1) / kThreads;
        GatherKernel << <static_cast<unsigned int>(blocks), kThreads, 0, stream >> > (data, indices, y, shape);
        CUDA_CHECK(cudaGetLastError());
    }
}