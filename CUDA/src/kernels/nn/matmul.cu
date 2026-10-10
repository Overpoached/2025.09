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
    // a: [m, k]
    // b: [k, n]
    // a * b = c
    // c: [m, n]
    __global__ void MatMulKernal(const float* __restrict__ a, 
        const float* __restrict__ b,
        const float* __restrict__ bias,
        float* c,
        GEMMShape shape)
    {
        __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
        __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
        int bx = blockIdx.x; int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;
        int Row = by * blockDim.y + ty;
        int Col = bx * blockDim.x + tx;
        int Batch = blockIdx.z;
        float Pvalue = 0;

        if (Batch >= shape.batch) return;

        // Loop over the M and N tiles required to compute the P element
        for (int p = 0; p < (shape.k - 1) / TILE_WIDTH + 1; ++p) {
            // Collaborative loading of M and N tiles into shared memory
            if (Row < shape.m && p * TILE_WIDTH + tx < shape.k) {
                ds_M[ty][tx] = a[Batch * shape.batchStrideA + Row * shape.k + p * TILE_WIDTH + tx];
            }
            else {
                ds_M[ty][tx] = 0.0;
            }
            if (!shape.transB && p * TILE_WIDTH + ty < shape.k && Col < shape.n) {
                ds_N[ty][tx] = b[Batch * shape.batchStrideB + (p * TILE_WIDTH + ty) * shape.n + Col];
            } else if (shape.transB && p * TILE_WIDTH + ty < shape.k && Col < shape.n) {
				ds_N[ty][tx] = b[Batch * shape.batchStrideB + Col * shape.k + (p * TILE_WIDTH + ty)];
			}
            else {
                ds_N[ty][tx] = 0.0;
            }
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; ++i) {
				Pvalue += ds_M[ty][i] * ds_N[i][tx];
            }
            __syncthreads();
        }


        if (Row < shape.m && Col < shape.n) {
			//if (shape.hasC) {
			//	Pvalue += bias[Batch * shape.batchStrideBias + Row * shape.n + Col];
			//}

            c[Batch * shape.batchStrideC + Row * shape.n + Col] = Pvalue;
        }
    }
    void GEMM(const float* a, const float* b, const float* bias, float* c,
        const GEMMShape& shape, cudaStream_t stream)
    {
        if (shape.m == 0 || shape.n == 0 || shape.k == 0) {
            return;
        }
        constexpr int kThreads = TILE_WIDTH;   // multiple of 32 (BlockSum requires it);
        dim3 grid((shape.n + kThreads - 1) / kThreads,
            (shape.m + kThreads - 1) / kThreads,
            shape.batch);
        dim3 block(kThreads, kThreads, 1);
        MatMulKernal << <grid, block, 0, stream >> > (a, b, bias, c, shape);
        
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