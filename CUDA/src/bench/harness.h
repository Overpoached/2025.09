#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bench/cuda_check.h"
#include "bench/cuda_timer.h"
#include "bench/gpu_state.h"
#include "bench/stats.h"

namespace bench {

// One thing to measure. `bytes_moved` and `flops` describe a single iteration
// and are what turn a duration into a rate; leave them zero when they do not
// apply and the corresponding rate is simply not reported.
struct BenchConfig {
    std::string name;           // what is measured, e.g. "vector_add"
    std::string variant;        // which implementation, e.g. "v1_scalar"
    std::string configuration;  // problem size, e.g. "n=67108864"
    int warmup_iters = 25;
    int measure_iters = 200;
    std::uint64_t bytes_moved = 0;
    std::uint64_t flops = 0;
    int device_index = 0;
};

struct BenchResult {
    BenchConfig config;
    Stats time_ms;
    GpuSample state_before;
    GpuSample state_after;

    // Rates are derived from the median, not the mean or the minimum. The
    // minimum flatters, the mean is dragged by the tail.
    double AchievedBandwidthGBs() const;
    double AchievedGFlops() const;

    // True when the SM clock moved by more than `tolerance` across the measured
    // run, which means the samples were not all taken under the same
    // conditions. With clocks locked this should never fire.
    bool ClocksDrifted(double tolerance = 0.05) const;

    // True when either sample saw a power/thermal limiter engaged.
    bool ThrottledDuringRun() const;
};

// Runs `launch` warmup_iters times untimed, then measure_iters times with a
// CUDA event pair around each individual call.
//
// Timing each iteration separately rather than wrapping the whole loop costs a
// synchronisation per iteration, but it is the only way to get a distribution -
// and a p99 is worth more than the microsecond of event overhead it costs.
// `launch` must enqueue work on the default stream and must not synchronise.
template <typename LaunchFn>
BenchResult RunBenchmark(const BenchConfig& config, LaunchFn&& launch) {
    for (int i = 0; i < config.warmup_iters; ++i) {
        launch();
    }
    CUDA_CHECK_KERNEL();

    BenchResult result;
    result.config = config;
    result.state_before = SampleGpuState(config.device_index);

    CudaEventTimer timer;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(config.measure_iters));
    for (int i = 0; i < config.measure_iters; ++i) {
        timer.Start();
        launch();
        timer.Stop();
        samples.push_back(timer.ElapsedMs());  // synchronises on the stop event
    }
    CUDA_CHECK(cudaGetLastError());

    result.state_after = SampleGpuState(config.device_index);
    result.time_ms = ComputeStats(std::move(samples));
    return result;
}

}  // namespace bench

#endif  // BENCH_HARNESS_H
