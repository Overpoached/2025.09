#include "bench/stats.h"

#include <algorithm>
#include <cmath>

namespace bench {

double PercentileSorted(const std::vector<double>& sorted, double percent) {
    if (sorted.empty()) {
        return 0.0;
    }
    // Nearest-rank: the smallest value at or below which `percent` of the
    // sample falls. Chosen over interpolation because it always returns a
    // measurement that actually occurred.
    const double rank = std::ceil(percent / 100.0 * static_cast<double>(sorted.size()));
    std::size_t index = rank > 0.0 ? static_cast<std::size_t>(rank) - 1 : 0;
    index = std::min(index, sorted.size() - 1);
    return sorted[index];
}

Stats ComputeStats(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }

    std::sort(samples.begin(), samples.end());
    stats.n = samples.size();
    stats.min = samples.front();
    stats.max = samples.back();

    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }
    stats.mean = sum / static_cast<double>(stats.n);

    if (stats.n > 1) {
        double sum_sq_dev = 0.0;
        for (double value : samples) {
            const double dev = value - stats.mean;
            sum_sq_dev += dev * dev;
        }
        stats.stddev = std::sqrt(sum_sq_dev / static_cast<double>(stats.n - 1));
    }

    const std::size_t mid = stats.n / 2;
    stats.median = (stats.n % 2 == 0) ? 0.5 * (samples[mid - 1] + samples[mid]) : samples[mid];
    stats.p25 = PercentileSorted(samples, 25.0);
    stats.p75 = PercentileSorted(samples, 75.0);
    stats.p95 = PercentileSorted(samples, 95.0);
    stats.p99 = PercentileSorted(samples, 99.0);
    return stats;
}

}  // namespace bench
