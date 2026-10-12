#include "kernels/memory/vector_add.h"

namespace kernels {
namespace {

constexpr int kThreadsPerBlock = 256;

int BlocksFor(int elements) { return (elements + kThreadsPerBlock - 1) / kThreadsPerBlock; }

__global__ void FillPatternKernel(float* __restrict__ a, float* __restrict__ b, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<float>(idx & 1023);
        b[idx] = static_cast<float>((idx & 511) * 2);
    }
}

__global__ void VectorAddScalarKernel(const float* __restrict__ a, const float* __restrict__ b,
                                      float* __restrict__ c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Consecutive threads still touch consecutive addresses, so access stays fully
// coalesced; the difference is that each thread now issues one 16-byte access
// instead of one 4-byte access. A warp covers four 128-byte cache lines per
// instruction instead of one, so the same traffic is requested with a quarter
// of the load/store instructions and fewer of them are needed in flight to keep
// the memory system saturated.
__global__ void VectorAddFloat4Kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                                      float4* __restrict__ c, int n4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const float4 va = a[idx];
        const float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

}  // namespace

void LaunchFillPattern(float* a, float* b, int n, cudaStream_t stream) {
    FillPatternKernel<<<BlocksFor(n), kThreadsPerBlock, 0, stream>>>(a, b, n);
}

void LaunchVectorAddScalar(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    VectorAddScalarKernel<<<BlocksFor(n), kThreadsPerBlock, 0, stream>>>(a, b, c, n);
}

void LaunchVectorAddFloat4(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    const int n4 = n / 4;
    VectorAddFloat4Kernel<<<BlocksFor(n4), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b),
        reinterpret_cast<float4*>(c), n4);
}

}  // namespace kernels
