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
	};
	void Conv(const float* x,
				const float* w, 
				float* y,
				const ConvShape& shapes,
				const int64_t* pad,		//up, right, down, left
				int64_t npad,
				const int64_t* stride,	//row x col (h x w)
				int64_t nstride,
				cudaStream_t stream);
}