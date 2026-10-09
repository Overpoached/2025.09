#pragma once

#include <cuda_runtime.h>



namespace kernels {

	// elementwise operations: Add, Sub, Mul, Div, Pow, Min, Max, And, Or, Xor
	void Sigmoid(const float* x, float* y, int64_t n, cudaStream_t stream);
	void Add(const float* x, const float* y, float* z, int64_t n, cudaStream_t stream);

	// Convolution operation
	struct ConvShape {
		int64_t xbatch, xchannels, xrows, xcols;
		int64_t wcount, wchannels, wrows, wcols;
		int64_t ybatch, ychannels, yrows, ycols;
		int64_t uppad, leftpad, downpad, rightpad;	// ONNX order: [top, left, bottom, right]
		int64_t rowstride, colstride;				// [row, col]
	};
	void Conv(const float* x,
				const float* w, 
				float* y,
				const ConvShape& shapes,
				cudaStream_t stream);

	// Transpose
	constexpr int kMaxTransposeRank = 6;
	struct TransposeShape {
		int32_t rank;
		int64_t n;                                // total elements
		int64_t outDims[kMaxTransposeRank];       // output shape
		int64_t inStrides[kMaxTransposeRank];     // input stride for each OUTPUT axis: inStride[perm[i]]
	};
	void Transpose(const float* x, float* y, const TransposeShape& shape, cudaStream_t stream);
}