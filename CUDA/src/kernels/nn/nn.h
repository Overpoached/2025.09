#pragma once

#include <cuda_runtime.h>
#include <vector>


namespace kernels {

	// elementwise operations: Add, Sub, Mul, Div, Pow, Min, Max, And, Or, Xor
	void Sigmoid(const float* x, float* y, int64_t n, cudaStream_t stream);

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

	// Broadcast binary ops (numpy rules, as ONNX Add/Mul/Sub/Div)
	constexpr int kMaxBroadcastRank = 6;
	struct BroadcastShape {
		int32_t rank;
		int64_t n;                                   // output elements
		int64_t outDims[kMaxBroadcastRank];
		int64_t aStrides[kMaxBroadcastRank];         // 0 on axes where a is broadcast
		int64_t bStrides[kMaxBroadcastRank];         // 0 on axes where b is broadcast
	};
	// a and b must have compatible shape
	BroadcastShape MakeBroadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b, std::vector<int64_t>& outShape);
	void Add(const float* a, const float* b, float* y, const BroadcastShape& shape, cudaStream_t stream);
	void Add(const float* a, const float* b, float* y, int64_t n, cudaStream_t stream);   // same shapes
	void Mul(const float* a, const float* b, float* y, const BroadcastShape& shape, cudaStream_t stream);
	void Mul(const float* a, const float* b, float* y, int64_t n, cudaStream_t stream);   // same shapes

	// LayerNormalization over the last `cols` elements of each row. bias may be nullptr.
	void LayerNorm(const float* x, const float* scale, const float* bias, float* y,
		int64_t rows, int64_t cols, float epsilon, cudaStream_t stream);

	// General Matrix Multiplication
	struct GEMMShape {
		int64_t batch = 1;
		int64_t m, k, n;
		int64_t batchStrideA = 0, batchStrideB = 0, batchStrideC = 0, batchStrideBias = 0;   // elements between consecutive matrices; 0 = shared
		int64_t biasRowStride = 0, biasColStride = 0;   // C/bias broadcast into [M, N]: [N] -> (0,1), [M,1] -> (1,0), [M,N] -> (N,1), scalar -> (0,0)
		float alpha = 1.0f, beta = 1.0f;
		int32_t transA = 0, transB = 0, hasC = 0;
	};
	kernels::GEMMShape MakeMatMulShape(std::vector<int64_t> a, std::vector<int64_t> b,
		std::vector<int64_t>& outShape);
	void GEMM(const float* a, const float* b, const float* bias, float* c, 
		const GEMMShape& shape, cudaStream_t stream);

	// Gather: data viewed as [outer, axisDim, inner]; output is [outer, numIndices, inner]
	struct GatherShape {
		int64_t outer, axisDim, inner;
		int64_t numIndices;
		int64_t n;   // output elements = outer * numIndices * inner
	};
	void Gather(const float* data, const int64_t* indices, float* y, const GatherShape& shape, cudaStream_t stream);

	// Softmax along one axis. Input viewed as [outer, cols, inner]; each (outer, inner) pair is one row.
	struct SoftmaxShape {
		int64_t rows;    // outer * inner
		int64_t cols;    // size of the softmax axis
		int64_t inner;   // stride between consecutive elements of a row (1 when axis is last)
	};
	void Softmax(const float* x, float* y, const SoftmaxShape& shape, cudaStream_t stream);
}