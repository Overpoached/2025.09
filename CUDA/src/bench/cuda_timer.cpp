#include "bench/cuda_timer.h"

#include "bench/cuda_check.h"

namespace bench {

CudaEventTimer::CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

CudaEventTimer::~CudaEventTimer() {
    // Destructors do not throw; a failure here cannot be acted on anyway.
    if (start_ != nullptr) {
        cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
    }
}

void CudaEventTimer::Start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
}

void CudaEventTimer::Stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
}

double CudaEventTimer::ElapsedMs() const {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return static_cast<double>(milliseconds);
}

}  // namespace bench
