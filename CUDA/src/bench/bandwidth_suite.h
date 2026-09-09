#ifndef BENCH_BANDWIDTH_SUITE_H
#define BENCH_BANDWIDTH_SUITE_H

#include <iosfwd>
#include <vector>

#include "bench/harness.h"

namespace bench {

struct BandwidthSuiteOptions {
    // Element counts to sweep. Small sizes are dominated by launch overhead and
    // fit in L2; only the large ones measure DRAM bandwidth. The sweep is what
    // makes that visible rather than a footnote.
    std::vector<int> element_counts;
    int warmup_iters = 25;
    int measure_iters = 200;
    int device_index = 0;
};

std::vector<int> DefaultElementCounts();

// Allocates the buffers, verifies each kernel against the expected result
// before timing it, then measures. Throws CudaError on any device failure and
// std::runtime_error if a kernel produces a wrong result - a fast wrong kernel
// is never recorded.
std::vector<BenchResult> RunBandwidthSuite(const BandwidthSuiteOptions& options,
                                           std::ostream& log);

}  // namespace bench

#endif  // BENCH_BANDWIDTH_SUITE_H
