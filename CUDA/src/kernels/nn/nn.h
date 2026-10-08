#pragma once

#include <cuda_runtime.h>



namespace kernels {
	// Convolution operation
	void Conv(const float* x, int64_t xrows, int64_t xcols, 
				const float* w, int64_t wrows, int64_t wcols, 
				float* y, int64_t yrows, int64_t ycols,
				const int64_t* pad,		//up, right, down, left
				int64_t npad,
				const int64_t* stride,	//row x col (h x w)
				int64_t nstride,
				cudaStream_t stream);
}