#include "bench/gpu_state.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <sstream>

#include "bench/cuda_check.h"

namespace bench {
namespace {

// Owns the NVML session for the life of the process. NVML is optional: if it
// will not initialise we still benchmark, we just cannot report clock state.
class NvmlSession {
public:
    static NvmlSession& Instance() {
        static NvmlSession session;
        return session;
    }

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    bool GetHandle(int index, nvmlDevice_t* device) const {
        if (!ok_) {
            return false;
        }
        return nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(index), device) == NVML_SUCCESS;
    }

private:
    NvmlSession() {
        const nvmlReturn_t rc = nvmlInit_v2();
        if (rc == NVML_SUCCESS) {
            ok_ = true;
        } else {
            error_ = nvmlErrorString(rc);
        }
    }

    ~NvmlSession() {
        if (ok_) {
            nvmlShutdown();
        }
    }

    NvmlSession(const NvmlSession&) = delete;
    NvmlSession& operator=(const NvmlSession&) = delete;

    bool ok_ = false;
    std::string error_;
};

int GetAttribute(cudaDeviceAttr attr, int device_index) {
    int value = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, device_index));
    return value;
}

}  // namespace

double DeviceInfo::TheoreticalBandwidthGBs() const {
    const double clock_hz = static_cast<double>(memory_clock_khz) * 1000.0;
    const double bytes_per_transfer = static_cast<double>(memory_bus_bits) / 8.0;
    // x2 for double data rate.
    return clock_hz * 2.0 * bytes_per_transfer / 1.0e9;
}

std::string DeviceInfo::ComputeCapability() const {
    std::ostringstream out;
    out << cc_major << "." << cc_minor;
    return out.str();
}

std::string DeviceInfo::ShortSummary() const {
    std::ostringstream out;
    out << name << " (sm_" << cc_major << cc_minor << ", " << sm_count << " SMs, "
        << memory_bus_bits << "-bit @ " << memory_clock_khz / 1000 << " MHz)";
    return out.str();
}

DeviceInfo QueryDeviceInfo(int device_index) {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_index));

    DeviceInfo info;
    info.index = device_index;
    info.name = props.name;
    info.cc_major = props.major;
    info.cc_minor = props.minor;
    info.sm_count = props.multiProcessorCount;
    info.global_mem_bytes = props.totalGlobalMem;
    info.l2_cache_bytes = props.l2CacheSize;
    info.memory_bus_bits = props.memoryBusWidth;
    info.memory_clock_khz = GetAttribute(cudaDevAttrMemoryClockRate, device_index);
    info.sm_clock_khz = GetAttribute(cudaDevAttrClockRate, device_index);

    nvmlDevice_t device{};
    if (NvmlSession::Instance().GetHandle(device_index, &device)) {
        unsigned value = 0;
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
            info.max_sm_clock_mhz = value;
        }
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
            info.max_mem_clock_mhz = value;
        }
    }
    return info;
}

GpuSample SampleGpuState(int device_index) {
    GpuSample sample;
    nvmlDevice_t device{};
    if (!NvmlSession::Instance().GetHandle(device_index, &device)) {
        return sample;
    }

    unsigned value = 0;
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
        sample.sm_clock_mhz = value;
    }
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
        sample.mem_clock_mhz = value;
    }
    // nvmlDeviceGetTemperature was deprecated in NVML 13.0 in favour of the
    // versioned struct form.
    nvmlTemperature_t temperature{};
    temperature.version = nvmlTemperature_v1;
    temperature.sensorType = NVML_TEMPERATURE_GPU;
    if (nvmlDeviceGetTemperatureV(device, &temperature) == NVML_SUCCESS) {
        sample.temperature_c = temperature.temperature;
    }
    // Power draw is not exposed on every consumer board; absence is normal.
    if (nvmlDeviceGetPowerUsage(device, &value) == NVML_SUCCESS) {
        sample.power_mw = value;
    }

    unsigned long long reasons = 0;
    if (nvmlDeviceGetCurrentClocksEventReasons(device, &reasons) == NVML_SUCCESS) {
        sample.clock_event_reasons = reasons;
    }

    sample.valid = true;
    return sample;
}

bool GpuSample::ThrottledForPerformanceReason() const {
    // GpuIdle means the clocks are low because there is no work, which is the
    // expected state before a run starts. ApplicationsClocksSetting means the
    // clocks are pinned, which is what we want during a benchmark. Neither
    // invalidates a measurement; everything below does.
    constexpr unsigned long long kPerformanceLimiters =
        nvmlClocksEventReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown |
        nvmlClocksEventReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown |
        nvmlClocksThrottleReasonHwPowerBrakeSlowdown;
    return (clock_event_reasons & kPerformanceLimiters) != 0;
}

std::string GpuSample::ThrottleDescription() const {
    if (!valid) {
        return "unknown";
    }
    if (clock_event_reasons == 0) {
        return "none";
    }

    struct ReasonBit {
        unsigned long long mask;
        const char* label;
    };
    static const ReasonBit kReasons[] = {
        {nvmlClocksEventReasonGpuIdle, "idle"},
        {nvmlClocksEventReasonApplicationsClocksSetting, "clocks-pinned"},
        {nvmlClocksEventReasonSwPowerCap, "sw-power-cap"},
        {nvmlClocksThrottleReasonHwSlowdown, "hw-slowdown"},
        {nvmlClocksEventReasonSyncBoost, "sync-boost"},
        {nvmlClocksEventReasonSwThermalSlowdown, "sw-thermal"},
        {nvmlClocksThrottleReasonHwThermalSlowdown, "hw-thermal"},
        {nvmlClocksThrottleReasonHwPowerBrakeSlowdown, "hw-power-brake"},
        {nvmlClocksEventReasonDisplayClockSetting, "display-clock"},
    };

    std::ostringstream out;
    bool first = true;
    for (const ReasonBit& reason : kReasons) {
        if ((clock_event_reasons & reason.mask) != 0) {
            if (!first) {
                out << ",";
            }
            out << reason.label;
            first = false;
        }
    }
    if (first) {
        out << "0x" << std::hex << clock_event_reasons;
    }
    return out.str();
}

bool NvmlAvailable() { return NvmlSession::Instance().ok(); }

std::string NvmlUnavailableReason() { return NvmlSession::Instance().error(); }

}  // namespace bench
