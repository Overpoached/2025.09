#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace kernels {

	// Pillow's uint8 resampler keeps weights in fixed point with this many fractional bits.
	constexpr int kResamplePrecisionBits = 32 - 8 - 2;

	// One axis of a separable resize, precomputed on the host.
	// Output pixel i reads count[i] source pixels starting at first[i],
	// weighted by coeffs[i * ksize + k] (fixed point, kResamplePrecisionBits).
	struct ResampleAxis {
		const int32_t* first;
		const int32_t* count;
		const int32_t* coeffs;
		int32_t ksize;
		int32_t outSize;
	};

	struct ImageNormalize {
		float mean[3];   // RGB order
		float std[3];
	};

	// Horizontal pass. src is [*, srcW, 3] uint8 (BGR, as OpenCV decodes),
	// rows [rowStart, rowStart + rows) are resized to dst [rows, axis.outSize, 3] uint8.
	void ResizeRows(const uint8_t* src, int srcW, int rowStart, int rows,
		uint8_t* dst, const ResampleAxis& axis, cudaStream_t stream);

	// Vertical pass fused with BGR->RGB, /255, (x - mean) / std and HWC->CHW.
	// tmp is [*, width, 3] uint8, axis.first is relative to tmp row 0.
	// dst is [3, axis.outSize, width] float.
	void ResizeColsNormalize(const uint8_t* tmp, int width, const ResampleAxis& axis,
		const ImageNormalize& norm, float* dst, cudaStream_t stream);
}