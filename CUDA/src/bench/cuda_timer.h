#ifndef BENCH_CUDA_TIMER_H
#define BENCH_CUDA_TIMER_H

#include <cuda_runtime.h>

namespace bench {

// Times GPU work with CUDA events.
//
// Host wall-clock timing measures launch overhead and driver queuing, not the
// kernel - launches are asynchronous, so a std::chrono pair around a launch
// mostly measures how fast the CPU got back from the driver. Events are
// recorded into the stream and timestamped by the GPU itself. Resolution is
// roughly half a microsecond.
class CudaEventTimer {
public:
    CudaEventTimer();
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void Start(cudaStream_t stream = nullptr);
    void Stop(cudaStream_t stream = nullptr);

    // Blocks until the stop event has been observed, then returns the interval
    // in milliseconds.
    double ElapsedMs() const;

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace bench

#endif  // BENCH_CUDA_TIMER_H
