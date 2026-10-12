#include "bench/harness.h"

#include <cmath>

namespace bench {

double BenchResult::AchievedBandwidthGBs() const {
    if (config.bytes_moved == 0 || time_ms.median <= 0.0) {
        return 0.0;
    }
    const double seconds = time_ms.median / 1000.0;
    return static_cast<double>(config.bytes_moved) / seconds / 1.0e9;
}

double BenchResult::AchievedGFlops() const {
    if (config.flops == 0 || time_ms.median <= 0.0) {
        return 0.0;
    }
    const double seconds = time_ms.median / 1000.0;
    return static_cast<double>(config.flops) / seconds / 1.0e9;
}

bool BenchResult::ClocksDrifted(double tolerance) const {
    if (!state_before.valid || !state_after.valid) {
        return false;
    }
    if (state_before.sm_clock_mhz == 0) {
        return false;
    }
    const double before = static_cast<double>(state_before.sm_clock_mhz);
    const double after = static_cast<double>(state_after.sm_clock_mhz);
    return std::fabs(after - before) / before > tolerance;
}

bool BenchResult::ThrottledDuringRun() const {
    return state_before.ThrottledForPerformanceReason() ||
           state_after.ThrottledForPerformanceReason();
}

}  // namespace bench
