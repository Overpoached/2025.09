#ifndef BENCH_GPU_STATE_H
#define BENCH_GPU_STATE_H

#include <cstddef>
#include <optional>
#include <string>

namespace bench {

// Static description of the device, read once at startup.
//
// Note for CUDA 13: cudaDeviceProp no longer carries clockRate or
// memoryClockRate - they were removed, not merely deprecated. The clock rates
// below come from cudaDeviceGetAttribute instead.
struct DeviceInfo {
    int index = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t global_mem_bytes = 0;
    int l2_cache_bytes = 0;
    int memory_bus_bits = 0;
    int memory_clock_khz = 0;  // peak, from cudaDevAttrMemoryClockRate
    int sm_clock_khz = 0;      // peak, from cudaDevAttrClockRate

    // NVML's view of the peak clocks. Zero when NVML is unavailable.
    unsigned max_sm_clock_mhz = 0;
    unsigned max_mem_clock_mhz = 0;

    // Peak theoretical bandwidth: memory clock x 2 (DDR transfers per cycle)
    // x bus width. This is the ceiling every bandwidth-bound kernel in this
    // project is measured against. No kernel reaches it; 70-85% is the
    // realistic band on this card.
    double TheoreticalBandwidthGBs() const;

    std::string ComputeCapability() const;  // "8.9"
    std::string ShortSummary() const;
};

// A point-in-time reading of clock and thermal state, sampled either side of
// every measured run so a benchmark entry records the conditions that produced
// it. Diagnostic only: an unavailable reading never fails a benchmark.
struct GpuSample {
    bool valid = false;
    unsigned sm_clock_mhz = 0;
    unsigned mem_clock_mhz = 0;
    int temperature_c = 0;
    std::optional<unsigned> power_mw;  // unsupported on many consumer cards
    unsigned long long clock_event_reasons = 0;

    // Human-readable decode of the NVML clock-event reason bitmask.
    std::string ThrottleDescription() const;

    // True when the clocks are being held down by power, thermal or hardware
    // limits - as opposed to simply being idle, or pinned by a deliberate
    // clock lock. A benchmark taken while this is true is not comparable to
    // one taken while it is false.
    bool ThrottledForPerformanceReason() const;
};

DeviceInfo QueryDeviceInfo(int device_index);

// NVML is initialised on first use and shut down at process exit. Returns an
// invalid sample rather than throwing when NVML is unavailable.
GpuSample SampleGpuState(int device_index);

bool NvmlAvailable();

// Human-readable reason NVML could not be used, empty when it is fine.
std::string NvmlUnavailableReason();

}  // namespace bench

#endif  // BENCH_GPU_STATE_H
