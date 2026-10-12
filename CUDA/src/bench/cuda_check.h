#ifndef BENCH_CUDA_CHECK_H
#define BENCH_CUDA_CHECK_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace bench {

// Thrown by CUDA_CHECK on any failed runtime call. Nothing in the harness
// attempts recovery; the runner catches this at the top level and reports.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

[[noreturn]] void ThrowCudaError(cudaError_t code, const char* expr, const char* file, int line);

inline void CheckCuda(cudaError_t code, const char* expr, const char* file, int line) {
    if (code != cudaSuccess) {
        ThrowCudaError(code, expr, file, line);
    }
}

}  // namespace bench

#define CUDA_CHECK(expr) ::bench::CheckCuda((expr), #expr, __FILE__, __LINE__)

// Kernel launches report failure in two places: cudaGetLastError() catches a
// bad launch configuration immediately, while a fault raised during execution
// only surfaces on the next synchronisation. Both are checked here, so this is
// a synchronising macro and must not appear inside a timed loop.
#define CUDA_CHECK_KERNEL()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

#endif  // BENCH_CUDA_CHECK_H
