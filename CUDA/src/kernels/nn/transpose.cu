#include "nn.h"

#include "../kernel_utility.h"

namespace kernels {

    __global__ void Transpose(const float* __restrict__ x,
        float* __restrict__ y,
        TransposeShape shape)
    {
        int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;   // linear OUTPUT index
        if (idx >= shape.n) {
            return;
        }

        // Peel off output coordinates from the innermost axis outwards, accumulating the input offset.
        int64_t rem = idx;
        int64_t src = 0;
        for (int d = shape.rank - 1; d >= 0; --d) {
            const int64_t coord = rem % shape.outDims[d];
            rem /= shape.outDims[d];
            src += coord * shape.inStrides[d];
        }

        y[idx] = x[src];   // write is coalesced, read is strided
    }

	void Transpose(const float* x, float* y, const TransposeShape& shape, cudaStream_t stream)
	{
		if (shape.n == 0) {
			return;
		}
		constexpr int kThreads = 256;
		const int64_t blocks = (shape.n + kThreads - 1) / kThreads;
		Transpose << <static_cast<unsigned int>(blocks), kThreads, 0, stream >> > (x, y, shape);
		CUDA_CHECK(cudaGetLastError());           // catches bad launch configs immediately
	}
}