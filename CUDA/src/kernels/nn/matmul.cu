#include "nn.h"

#include "../kernel_utility.h"
#define TILE_WIDTH 16

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
	// C[m][n] = alpha * sum_k op(A)[m][k] * op(B)[k][n]  +  beta * bias[m * biasRowStride + n * biasColStride]
	//   op(A) = A    stored [M, K]   (TransA = false)
	//         = A^T  stored [K, M]   (TransA = true)
	//   op(B) = B    stored [K, N]   (TransB = false)
	//         = B^T  stored [N, K]   (TransB = true)
	// Shared tiles are always op(A)[row][k] and op(B)[k][col], so the inner product loop is the same
	// for all four variants; only the global->shared loads differ.
	template <bool TransA, bool TransB>
    __global__ void MatMulKernel(const float* __restrict__ a, 
        const float* __restrict__ b,
        const float* __restrict__ bias,
        float* c,
        GEMMShape shape)
    {
        __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH + 1];		// op(A) tile: [row][k]
        __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH + 1];		// op(B) tile: [k][col]
        const int tx = threadIdx.x; const int ty = threadIdx.y;
		const int64_t rowBase = int64_t(blockIdx.y) * TILE_WIDTH;
		const int64_t colBase = int64_t(blockIdx.x) * TILE_WIDTH;
		const int64_t row = rowBase + ty;
		const int64_t col = colBase + tx;
		const int64_t Batch = blockIdx.z;

        if (Batch >= shape.batch) return;

		const float* A = a + Batch * shape.batchStrideA;
		const float* B = b + Batch * shape.batchStrideB;
		const int64_t M = shape.m, N = shape.n, K = shape.k;

		float Pvalue = 0.0f;
        // Loop over the M and N tiles required to compute the P element
        for (int64_t k0 = 0; k0 < K; k0 += TILE_WIDTH) {
            // Collaborative loading of M and N tiles into shared memory
			// ---- op(A) tile ----
			if (!TransA) {
				// A is [M, K]: tx walks k, which is contiguous -> coalesced
				const int64_t k = k0 + tx;
				ds_M[ty][tx] = (row < M && k < K) ? A[row * K + k] : 0.0f;
			}
			else {
				// A is [K, M]: tx walks m (contiguous), then store transposed into [row][k]
				const int64_t k = k0 + ty;
				const int64_t m = rowBase + tx;
				ds_M[tx][ty] = (m < M && k < K) ? A[k * M + m] : 0.0f;
			}

			// ---- op(B) tile ----
			if (!TransB) {
				// B is [K, N]: tx walks n, which is contiguous -> coalesced
				const int64_t k = k0 + ty;
				ds_N[ty][tx] = (k < K && col < N) ? B[k * N + col] : 0.0f;
			}
			else {
				// B is [N, K]: tx walks k (contiguous), then store transposed into [k][col]
				const int64_t n = colBase + ty;
				const int64_t k = k0 + tx;
				ds_N[tx][ty] = (n < N && k < K) ? B[n * K + k] : 0.0f;
			}
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; ++i) {
				Pvalue += ds_M[ty][i] * ds_N[i][tx];
            }
            __syncthreads();
        }

		// ---- epilogue: alpha, bias (beta * C), store ----
		if (row < M && col < N) {
			float v = shape.alpha * Pvalue;
			if (bias) {
				v += shape.beta * bias[Batch * shape.batchStrideBias + row * shape.biasRowStride + col * shape.biasColStride];
			}
			c[Batch * shape.batchStrideC + row * N + col] = v;
		}
    }
    void GEMM(const float* a, const float* b, const float* bias, float* c,
        const GEMMShape& shape, cudaStream_t stream)
    {
        if (shape.m == 0 || shape.n == 0 || shape.batch == 0) {
            return; // K == 0 is allowed: the result is just the epilogue (alpha*0 + beta*bias)
        }
		const dim3 block(TILE_WIDTH, TILE_WIDTH, 1);   // 16x16 = 256 threads
		const dim3 grid(static_cast<unsigned int>((shape.n + TILE_WIDTH - 1) / TILE_WIDTH),
			static_cast<unsigned int>((shape.m + TILE_WIDTH - 1) / TILE_WIDTH),
			static_cast<unsigned int>(shape.batch));
       
		// One compiled variant per layout: no runtime branches in the main loop
		if (!shape.transA && !shape.transB) {
			MatMulKernel<false, false> << <grid, block, 0, stream >> > (a, b, bias, c, shape);
		}
		else if (!shape.transA && shape.transB) {
			MatMulKernel<false, true> << <grid, block, 0, stream >> > (a, b, bias, c, shape);
		}
		else if (shape.transA && !shape.transB) {
			MatMulKernel<true, false> << <grid, block, 0, stream >> > (a, b, bias, c, shape);
		}
		else {
			MatMulKernel<true, true> << <grid, block, 0, stream >> > (a, b, bias, c, shape);
		}
		CUDA_CHECK(cudaGetLastError());
    }

	// ONNX MatMul = numpy.matmul:
		//   last two dims are the matrices: [..., M, K] x [..., K, N] -> [..., M, N]
		//   leading (batch) dims broadcast with numpy rules
		//   1-D A is treated as [1, K] and 1-D B as [K, 1]; that dim is dropped from the output
	kernels::GEMMShape MakeMatMulShape(std::vector<int64_t> a, std::vector<int64_t> b,
		std::vector<int64_t>& outShape)
	{
		if (a.empty() || b.empty()) {
			throw std::runtime_error("MatMul: scalar inputs not allowed");
		}
		const bool aVec = a.size() == 1;
		const bool bVec = b.size() == 1;
		if (aVec) a.insert(a.begin(), 1);   // [K]   -> [1, K]
		if (bVec) b.push_back(1);           // [K]   -> [K, 1]

		const int64_t M = a[a.size() - 2];
		const int64_t K = a.back();
		const int64_t N = b.back();
		if (b[b.size() - 2] != K) {
			throw std::runtime_error("MatMul: inner dims differ " + ToString(a) + " x "
				+ ToString(b));
		}

		// Batch dims: everything before the last two, right-aligned and padded with 1s
		std::vector<int64_t> aBatch(a.begin(), a.end() - 2);
		std::vector<int64_t> bBatch(b.begin(), b.end() - 2);
		const size_t rank = std::max(aBatch.size(), bBatch.size());
		aBatch.insert(aBatch.begin(), rank - aBatch.size(), 1);
		bBatch.insert(bBatch.begin(), rank - bBatch.size(), 1);

		std::vector<int64_t> batchShape(rank);
		int64_t batch = 1, aCount = 1, bCount = 1;
		for (size_t d = 0; d < rank; ++d) {
			if (aBatch[d] != bBatch[d] && aBatch[d] != 1 && bBatch[d] != 1) {
				throw std::runtime_error("MatMul: incompatible batch dims " + ToString(a) + " x "
					+ ToString(b));
			}
			batchShape[d] = std::max(aBatch[d], bBatch[d]);
			batch *= batchShape[d];
			aCount *= aBatch[d];
			bCount *= bBatch[d];
		}

		kernels::GEMMShape s;
		s.k = K;
		s.n = N;
		if (bCount == 1) {
			// One B for everything (activation x weight): fold A's batch into M -> a single 2-D GEMM.
			// Valid because A and C are contiguous row-major, so [..., M, K] is [batch*M, K] in memory.
			s.batch = 1;
			s.m = batch * M;
			s.batchStrideA = s.m * K;
			s.batchStrideB = 0;
			s.batchStrideC = s.m * N;
		}
		else if (aCount == batch && bCount == batch) {
			// Same batch dims on both sides (attention): one matrix pair per batch index
			s.batch = batch;
			s.m = M;
			s.batchStrideA = M * K;
			s.batchStrideB = K * N;
			s.batchStrideC = M * N;
		}
		else if (aCount == 1) {
			// One A shared across all of B's batches
			s.batch = batch;
			s.m = M;
			s.batchStrideA = 0;
			s.batchStrideB = K * N;
			s.batchStrideC = M * N;
		}
		else {
			// e.g. [4,1,M,K] x [1,12,K,N]: needs per-dim batch strides. CLIP never does this.
			throw std::runtime_error("MatMul: partial batch broadcast not supported " + ToString(a)
				+ " x " + ToString(b));
		}

		outShape = batchShape;
		if (!aVec) outShape.push_back(M);
		if (!bVec) outShape.push_back(N);
		return s;
	}
}