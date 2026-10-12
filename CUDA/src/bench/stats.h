#ifndef BENCH_STATS_H
#define BENCH_STATS_H

#include <cstddef>
#include <vector>

namespace bench {

// Summary of a sample of per-iteration timings.
//
// Every performance number this project reports is a distribution, never a
// single measurement. The median is the headline; the tail is where WDDM
// scheduling, clock drift and thermal behaviour become visible.
struct Stats {
    std::size_t n = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;  // sample standard deviation, N-1 denominator
    double p25 = 0.0;
    double median = 0.0;
    double p75 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;

    // Coefficient of variation: stddev relative to mean. Comparable across
    // kernels of very different runtimes, but it is built on the mean and the
    // standard deviation, so a handful of extreme outliers dominate it.
    double cv() const { return mean > 0.0 ? stddev / mean : 0.0; }

    // Interquartile range over the median: the same idea as cv(), but built
    // from order statistics so outliers cannot move it.
    //
    // This is the trust metric that matters under WDDM. Windows will
    // occasionally deschedule the GPU mid-run and inject a stall of several
    // milliseconds into one iteration. That wrecks cv() while leaving the
    // typical iteration - and therefore the median we actually report -
    // perfectly reproducible. Judging the environment by cv() alone would
    // condemn a machine that is measuring fine.
    double robust_cv() const { return median > 0.0 ? (p75 - p25) / median : 0.0; }

    // How far the tail runs past the typical iteration. High tail_ratio with a
    // low robust_cv() is the WDDM signature: a tight bulk plus rare stalls.
    double tail_ratio() const { return median > 0.0 ? p99 / median : 0.0; }
};

// Nearest-rank percentile over an already-sorted sample. `percent` in (0, 100].
double PercentileSorted(const std::vector<double>& sorted, double percent);

// Takes the sample by value: it is sorted internally.
Stats ComputeStats(std::vector<double> samples);

}  // namespace bench

#endif  // BENCH_STATS_H
