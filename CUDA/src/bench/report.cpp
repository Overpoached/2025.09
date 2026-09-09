#include "bench/report.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace bench {
namespace {

std::string FormatBytes(std::size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (bytes >= (1ull << 30)) {
        out << static_cast<double>(bytes) / (1ull << 30) << " GiB";
    } else {
        out << static_cast<double>(bytes) / (1ull << 20) << " MiB";
    }
    return out.str();
}

// A configuration whose whole working set is comparable to the L2 is not
// measuring DRAM bandwidth: some or all of the traffic is served by cache, and
// the resulting "percentage of theoretical" is meaningless or above 100%.
// Marked rather than hidden - the contrast between the cached and uncached ends
// of the sweep is the most instructive thing in the table.
bool IsL2Influenced(const BenchResult& result, const DeviceInfo& device) {
    if (device.l2_cache_bytes <= 0 || result.config.bytes_moved == 0) {
        return false;
    }
    return result.config.bytes_moved < 2ull * static_cast<std::uint64_t>(device.l2_cache_bytes);
}

std::string Fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string Percent(double fraction, int precision) { return Fixed(100.0 * fraction, precision) + "%"; }

std::string DescribeClockState(const GpuSample& state) {
    if (!state.valid) {
        return "unavailable (NVML: " + NvmlUnavailableReason() + ")";
    }
    std::ostringstream out;
    out << "SM " << state.sm_clock_mhz << " MHz, MEM " << state.mem_clock_mhz << " MHz, "
        << state.temperature_c << " C";
    if (state.power_mw.has_value()) {
        out << ", " << std::fixed << std::setprecision(1)
            << static_cast<double>(*state.power_mw) / 1000.0 << " W";
    } else {
        out << ", power n/a";
    }
    out << ", limiters: " << state.ThrottleDescription();
    return out.str();
}

}  // namespace

bool IsDebugBuild() {
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
}

std::string BuildInfo() {
    std::ostringstream out;
    out << (IsDebugBuild() ? "Debug" : "Release") << " x64, MSVC " << _MSC_VER
        << ", CUDA runtime " << CUDART_VERSION / 1000 << "." << (CUDART_VERSION % 1000) / 10;
    return out.str();
}

std::string LocalDateStamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &local);
    return buffer;
}

void PrintDeviceBanner(std::ostream& out, const DeviceInfo& device, const GpuSample& state) {
    out << "Device " << device.index << ": " << device.name << "\n"
        << "  compute capability : " << device.ComputeCapability() << " (sm_" << device.cc_major
        << device.cc_minor << ")\n"
        << "  SMs                : " << device.sm_count << "\n"
        << "  global memory      : " << FormatBytes(device.global_mem_bytes) << "\n"
        << "  L2 cache           : " << FormatBytes(static_cast<std::size_t>(device.l2_cache_bytes))
        << "\n"
        << "  memory bus         : " << device.memory_bus_bits << "-bit @ "
        << device.memory_clock_khz / 1000 << " MHz\n"
        << "  theoretical BW     : " << std::fixed << std::setprecision(1)
        << device.TheoreticalBandwidthGBs() << " GB/s\n"
        << "  clock state        : " << DescribeClockState(state) << "\n"
        << "  build              : " << BuildInfo() << "\n";
}

void PrintResultsTable(std::ostream& out, const std::vector<BenchResult>& results,
                       const DeviceInfo& device) {
    const double peak = device.TheoreticalBandwidthGBs();

    out << "\n"
        << std::left << std::setw(13) << "variant" << std::setw(15) << "configuration"
        << std::right << std::setw(11) << "median ms" << std::setw(10) << "p99 ms"
        << std::setw(10) << "min ms" << std::setw(10) << "GB/s" << std::setw(9) << "% peak"
        << std::setw(9) << "IQR/med" << std::setw(9) << "p99/med"
        << "\n"
        << std::string(95, '-') << "\n";

    bool any_cached = false;
    for (const BenchResult& result : results) {
        const Stats& time = result.time_ms;
        const double gbps = result.AchievedBandwidthGBs();
        const bool cached = IsL2Influenced(result, device);
        any_cached = any_cached || cached;

        out << std::left << std::setw(13) << result.config.variant << std::setw(15)
            << (result.config.configuration + (cached ? " *" : "")) << std::right << std::setw(11)
            << Fixed(time.median, 4) << std::setw(10) << Fixed(time.p99, 4) << std::setw(10)
            << Fixed(time.min, 4) << std::setw(10) << Fixed(gbps, 1) << std::setw(9)
            << Percent(peak > 0.0 ? gbps / peak : 0.0, 1) << std::setw(9)
            << Percent(time.robust_cv(), 1) << std::setw(9) << Fixed(time.tail_ratio(), 2) << "\n";
    }

    if (!results.empty()) {
        out << "\n(rates derived from the median; " << results.front().config.measure_iters
            << " timed iterations after " << results.front().config.warmup_iters << " warmup)\n"
            << "IQR/med is the interquartile range over the median - dispersion of the bulk of\n"
               "the sample, immune to outliers. p99/med is how far the tail runs past typical.\n";
    }
    if (any_cached) {
        out << "* working set is within 2x the " << device.l2_cache_bytes / (1024 * 1024)
            << " MiB L2, so this row is served partly or wholly from cache and does not\n"
               "  measure DRAM bandwidth. The percentage of theoretical is not meaningful "
               "there.\n";
    }
}

bool PrintMeasurementWarnings(std::ostream& out, const std::vector<BenchResult>& results) {
    bool trustworthy = true;
    int tail_heavy = 0;

    if (IsDebugBuild()) {
        out << "\n[WARNING] Debug build. Device code is compiled with -G, which disables "
               "essentially\n          all optimisation. Do not record these numbers.\n";
        trustworthy = false;
    }

    if (!NvmlAvailable()) {
        out << "\n[WARNING] NVML unavailable (" << NvmlUnavailableReason()
            << ").\n          Clock and thermal state could not be recorded.\n";
        trustworthy = false;
    }

    for (const BenchResult& result : results) {
        const std::string label = result.config.variant + " " + result.config.configuration;
        if (result.ThrottledDuringRun()) {
            out << "\n[WARNING] " << label << ": clocks were limited during the run ("
                << result.state_after.ThrottleDescription() << ").\n";
            trustworthy = false;
        }
        if (result.ClocksDrifted()) {
            out << "\n[WARNING] " << label << ": SM clock moved from "
                << result.state_before.sm_clock_mhz << " to " << result.state_after.sm_clock_mhz
                << " MHz across the run.\n          Lock the clocks with tools/lock_clocks.ps1.\n";
            trustworthy = false;
        }
        if (result.time_ms.robust_cv() > 0.05) {
            out << "\n[WARNING] " << label << ": interquartile range is "
                << Percent(result.time_ms.robust_cv(), 1)
                << " of the median.\n          Above ~5% the bulk of the sample is too spread out "
                   "to trust.\n";
            trustworthy = false;
        } else if (result.time_ms.tail_ratio() > 1.5) {
            ++tail_heavy;
        }
    }

    if (tail_heavy > 0) {
        // Deliberately a note and not a warning: it does not clear the
        // trustworthy flag. A tight interquartile range with a long tail means
        // the typical iteration is reproducible and something outside the
        // kernel occasionally interrupts it.
        out << "\n[NOTE] " << tail_heavy
            << " configuration(s) had a tight bulk but a long tail (p99 over 1.5x the "
               "median).\n       Under WDDM the OS periodically deschedules the GPU mid-run. "
               "That inflates\n       p99 and the standard deviation without moving the median. "
               "Treat those p99s as\n       a property of the platform, not of the kernel.\n";
    }

    if (trustworthy) {
        out << "\nMeasurement environment looks sound: Release build, clocks steady, no "
               "throttling,\nbulk-of-sample dispersion within tolerance.\n";
    }
    return trustworthy;
}

std::string MakeMarkdownEntry(const std::vector<BenchResult>& results, const DeviceInfo& device,
                              const GpuSample& state, const std::string& title,
                              const std::string& what_changed) {
    const double peak = device.TheoreticalBandwidthGBs();
    std::ostringstream out;

    out << "## " << LocalDateStamp() << " - " << title << "\n\n";
    out << "**What changed:** " << what_changed << "\n\n";
    out << "**Hardware:** " << device.ShortSummary() << ", "
        << FormatBytes(device.global_mem_bytes) << " global, "
        << FormatBytes(static_cast<std::size_t>(device.l2_cache_bytes))
        << " L2. Theoretical bandwidth " << std::fixed << std::setprecision(1) << peak
        << " GB/s.\n\n";
    out << "**Clock state:** " << DescribeClockState(state) << "\n\n";
    out << "**Build:** " << BuildInfo() << ", sm_89, -lineinfo, no fast-math.\n\n";
    if (!results.empty()) {
        out << "**Method:** CUDA events around each individual launch. "
            << results.front().config.warmup_iters << " warmup iterations discarded, "
            << results.front().config.measure_iters
            << " timed iterations retained; rates derived from the median. Bytes moved counted "
               "as 2 reads + 1 write per element.\n\n";
    }

    out << "| variant | configuration | median (ms) | p99 (ms) | min (ms) | GB/s | % of peak | "
           "IQR/med | p99/med |\n";
    out << "|---|---|---:|---:|---:|---:|---:|---:|---:|\n";
    bool any_cached = false;
    for (const BenchResult& result : results) {
        const Stats& time = result.time_ms;
        const double gbps = result.AchievedBandwidthGBs();
        const bool cached = IsL2Influenced(result, device);
        any_cached = any_cached || cached;
        out << "| `" << result.config.variant << "` | " << result.config.configuration
            << (cached ? " \\*" : "") << " | " << Fixed(time.median, 4) << " | "
            << Fixed(time.p99, 4) << " | " << Fixed(time.min, 4) << " | " << Fixed(gbps, 1)
            << " | " << Percent(peak > 0.0 ? gbps / peak : 0.0, 1) << " | "
            << Percent(time.robust_cv(), 1) << " | " << Fixed(time.tail_ratio(), 2) << " |\n";
    }
    if (any_cached) {
        out << "\n\\* Working set is within 2x the " << device.l2_cache_bytes / (1024 * 1024)
            << " MiB L2, so the row is served partly or wholly from cache and does not measure "
               "DRAM bandwidth; its percentage of theoretical is not meaningful.\n";
    }
    out << "\n";
    return out.str();
}

}  // namespace bench
