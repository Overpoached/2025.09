#ifndef BENCH_REPORT_H
#define BENCH_REPORT_H

#include <iosfwd>
#include <string>
#include <vector>

#include "bench/gpu_state.h"
#include "bench/harness.h"

namespace bench {

// Compiler, toolkit and configuration the binary was built with. Recorded in
// every log entry, because a number produced by a Debug build is not comparable
// to one from a Release build and the difference is invisible six weeks later.
std::string BuildInfo();

// True when this binary was built without optimisation. Every performance
// number from such a build is meaningless.
bool IsDebugBuild();

std::string LocalDateStamp();  // YYYY-MM-DD

void PrintDeviceBanner(std::ostream& out, const DeviceInfo& device, const GpuSample& state);

void PrintResultsTable(std::ostream& out, const std::vector<BenchResult>& results,
                       const DeviceInfo& device);

// Warnings about the measurement environment: drifting clocks, thermal
// throttling, high variance, Debug build. Returns false when something was
// reported that makes the numbers untrustworthy.
bool PrintMeasurementWarnings(std::ostream& out, const std::vector<BenchResult>& results);

// A complete, dated BENCHMARKS.md section for this run, ready to append.
std::string MakeMarkdownEntry(const std::vector<BenchResult>& results, const DeviceInfo& device,
                              const GpuSample& state, const std::string& title,
                              const std::string& what_changed);

}  // namespace bench

#endif  // BENCH_REPORT_H
