#ifndef KERNELS_MEMORY_VECTOR_ADD_H
#define KERNELS_MEMORY_VECTOR_ADD_H

#include <cuda_runtime.h>

namespace kernels {

// c[i] = a[i] + b[i].
//
// One add per three memory accesses is an arithmetic intensity of 1/12 FLOP per
// byte, so this kernel does essentially no arithmetic and is bound purely by
// memory bandwidth. That makes it the right probe for establishing what
// fraction of theoretical bandwidth this machine can actually reach - the
// ceiling every later kernel is measured against.

// Fills a and b with the deterministic pattern that ExpectedSum() predicts, on
// the device, so verification needs no host-side copy of the inputs.
void LaunchFillPattern(float* a, float* b, int n, cudaStream_t stream = nullptr);

// v1: one float per thread. The straightforward version.
void LaunchVectorAddScalar(const float* a, const float* b, float* c, int n,
                           cudaStream_t stream = nullptr);

// v2: one float4 per thread. Identical memory traffic, a quarter of the memory
// instructions. Requires n % 4 == 0; pointers from cudaMalloc are always
// sufficiently aligned.
void LaunchVectorAddFloat4(const float* a, const float* b, float* c, int n,
                           cudaStream_t stream = nullptr);

// The value LaunchVectorAddScalar must produce at index i. Both operands are
// small integers exactly representable in float, so the expected result is
// exact and verification can use equality rather than a tolerance.
inline float ExpectedSum(int i) {
    return static_cast<float>(i & 1023) + static_cast<float>((i & 511) * 2);
}

}  // namespace kernels

#endif  // KERNELS_MEMORY_VECTOR_ADD_H
