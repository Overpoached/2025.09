#include "preprocess.h"

#include "../kernel_utility.h"

/*
* Resize + crop + normalise, matching Pillow's Image.resize(BICUBIC) bit for bit.
*
* A resize is separable: shrink every row (horizontal pass), then shrink every column
* of that result (vertical pass). Each output pixel is a weighted sum of a short run of
* input pixels. The weights depend only on the output coordinate, so the host computes
* them once per axis and the kernels just do the multiply-adds.
*
* Cropping is free: we only compute the 224 output columns/rows inside the crop window.
*/

namespace kernels {
	namespace {
		constexpr int32_t kHalf = 1 << (kResamplePrecisionBits - 1);   // rounding bias

		__device__ __forceinline__ uint8_t Clip8(int32_t s) {
			if (s >= (1 << kResamplePrecisionBits << 8)) return 255;
			if (s <= 0) return 0;
			return static_cast<uint8_t>(s >> kResamplePrecisionBits);
		}
	}

	// One thread per (output column, row). 3 channels per thread.
	__global__ void ResizeRowsKernel(const uint8_t* __restrict__ src, int srcW, int rowStart, int rows,
		uint8_t* __restrict__ dst, ResampleAxis axis)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int r = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= axis.outSize || r >= rows) return;

		const uint8_t* in = src + static_cast<int64_t>(rowStart + r) * srcW * 3;
		const int32_t* k = axis.coeffs + static_cast<int64_t>(x) * axis.ksize;
		const int first = axis.first[x];
		const int count = axis.count[x];

		int32_t s0 = kHalf, s1 = kHalf, s2 = kHalf;
		for (int i = 0; i < count; ++i) {
			const uint8_t* p = in + (first + i) * 3;
			s0 += p[0] * k[i];
			s1 += p[1] * k[i];
			s2 += p[2] * k[i];
		}

		uint8_t* out = dst + (static_cast<int64_t>(r) * axis.outSize + x) * 3;
		out[0] = Clip8(s0);
		out[1] = Clip8(s1);
		out[2] = Clip8(s2);
	}

	// One thread per output pixel. Reads BGR uint8, writes 3 planes of RGB float.
	__global__ void ResizeColsNormalizeKernel(const uint8_t* __restrict__ tmp, int width,
		ResampleAxis axis, ImageNormalize norm, float* __restrict__ dst)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= axis.outSize) return;

		const int32_t* k = axis.coeffs + static_cast<int64_t>(y) * axis.ksize;
		const int first = axis.first[y];
		const int count = axis.count[y];

		int32_t s[3] = { kHalf, kHalf, kHalf };
		for (int i = 0; i < count; ++i) {
			const uint8_t* p = tmp + (static_cast<int64_t>(first + i) * width + x) * 3;
			s[0] += p[0] * k[i];
			s[1] += p[1] * k[i];
			s[2] += p[2] * k[i];
		}

		const int64_t plane = static_cast<int64_t>(axis.outSize) * width;
		const int64_t pix = static_cast<int64_t>(y) * width + x;
		for (int c = 0; c < 3; ++c) {
			const int rgb = 2 - c;                                   // BGR -> RGB
			// Same op order as torchvision: x.div(255), then sub(mean).div(std). _rn = IEEE, no fast-math.
			const float v = __fdiv_rn(static_cast<float>(Clip8(s[c])), 255.0f);
			dst[rgb * plane + pix] = __fdiv_rn(__fsub_rn(v, norm.mean[rgb]), norm.std[rgb]);
		}
	}

	void ResizeRows(const uint8_t* src, int srcW, int rowStart, int rows,
		uint8_t* dst, const ResampleAxis& axis, cudaStream_t stream)
	{
		const dim3 block(32, 8);
		const dim3 grid((axis.outSize + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
		ResizeRowsKernel << <grid, block, 0, stream >> > (src, srcW, rowStart, rows, dst, axis);
		CUDA_CHECK_ERROR();
	}

	void ResizeColsNormalize(const uint8_t* tmp, int width, const ResampleAxis& axis,
		const ImageNormalize& norm, float* dst, cudaStream_t stream)
	{
		const dim3 block(32, 8);
		const dim3 grid((width + block.x - 1) / block.x, (axis.outSize + block.y - 1) / block.y);
		ResizeColsNormalizeKernel << <grid, block, 0, stream >> > (tmp, width, axis, norm, dst);
		CUDA_CHECK_ERROR();
	}
}