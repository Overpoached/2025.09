#include "bench/bandwidth_suite.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/cuda_check.h"
#include "bench/device_buffer.h"
#include "kernels/memory/vector_add.h"

namespace bench {
namespace {

// Verifying the whole array would mean copying back 256 MiB per configuration.
// The head catches indexing and offset mistakes, the tail catches the bounds
// guard, which is where this kind of kernel actually goes wrong.
constexpr int kVerifySampleElements = 1 << 20;

std::string DescribeSize(int n) {
    std::ostringstream out;
    if (n % (1 << 20) == 0) {
        out << "n=" << (n >> 20) << "Mi";
    } else if (n % (1 << 10) == 0) {
        out << "n=" << (n >> 10) << "Ki";
    } else {
        out << "n=" << n;
    }
    return out.str();
}

void VerifyRange(const DeviceBuffer<float>& c, int n, int offset, int count,
                 const std::string& label) {
    if (count <= 0) {
        return;
    }
    std::vector<float> host(static_cast<std::size_t>(count));
    c.CopyToHost(host.data(), host.size(), static_cast<std::size_t>(offset));

    for (int i = 0; i < count; ++i) {
        const int index = offset + i;
        const float expected = kernels::ExpectedSum(index);
        if (host[static_cast<std::size_t>(i)] != expected) {
            std::ostringstream out;
            out << label << " produced a wrong result at index " << index << " of " << n << ": got "
                << host[static_cast<std::size_t>(i)] << ", expected " << expected;
            throw std::runtime_error(out.str());
        }
    }
}

// Runs the kernel once into a poisoned output buffer and checks the result
// before any timing happens. Correctness first, always.
template <typename LaunchFn>
void VerifyKernel(DeviceBuffer<float>& c, int n, const std::string& label, LaunchFn&& launch) {
    c.Fill(0xFF);  // NaN-ish poison: an untouched element cannot compare equal
    launch();
    CUDA_CHECK_KERNEL();

    const int head = std::min(n, kVerifySampleElements);
    VerifyRange(c, n, 0, head, label);
    const int tail = std::min(n - head, kVerifySampleElements);
    if (tail > 0) {
        VerifyRange(c, n, n - tail, tail, label);
    }
}

}  // namespace

std::vector<int> DefaultElementCounts() {
    // 1Mi is 12 MiB of traffic and fits inside the 24 MiB L2; 64Mi is 768 MiB
    // and cannot. The pair of numbers is the point of the sweep.
    return {1 << 20, 1 << 22, 1 << 24, 1 << 26};
}

std::vector<BenchResult> RunBandwidthSuite(const BandwidthSuiteOptions& options,
                                           std::ostream& log) {
    std::vector<BenchResult> results;

    for (const int n : options.element_counts) {
        if (n <= 0 || n % 4 != 0) {
            throw std::runtime_error("element count must be positive and a multiple of 4: " +
                                     std::to_string(n));
        }

        const std::string size_label = DescribeSize(n);
        const std::uint64_t bytes_moved = 3ull * static_cast<std::uint64_t>(n) * sizeof(float);

        log << "  " << size_label << " (" << bytes_moved / (1024 * 1024)
            << " MiB moved per iteration): allocating, filling, verifying... ";
        log.flush();

        DeviceBuffer<float> a(static_cast<std::size_t>(n));
        DeviceBuffer<float> b(static_cast<std::size_t>(n));
        DeviceBuffer<float> c(static_cast<std::size_t>(n));

        kernels::LaunchFillPattern(a.data(), b.data(), n);
        CUDA_CHECK_KERNEL();

        struct Variant {
            const char* name;
            void (*launch)(const float*, const float*, float*, int, cudaStream_t);
        };
        const Variant variants[] = {
            {"v1_scalar", &kernels::LaunchVectorAddScalar},
            {"v2_float4", &kernels::LaunchVectorAddFloat4},
        };

        for (const Variant& variant : variants) {
            VerifyKernel(c, n, std::string(variant.name) + " " + size_label, [&] {
                variant.launch(a.data(), b.data(), c.data(), n, nullptr);
            });
        }
        log << "ok\n";
        log.flush();

        for (const Variant& variant : variants) {
            BenchConfig config;
            config.name = "vector_add";
            config.variant = variant.name;
            config.configuration = size_label;
            config.warmup_iters = options.warmup_iters;
            config.measure_iters = options.measure_iters;
            config.bytes_moved = bytes_moved;
            config.flops = static_cast<std::uint64_t>(n);  // one add per element
            config.device_index = options.device_index;

            results.push_back(RunBenchmark(config, [&] {
                variant.launch(a.data(), b.data(), c.data(), n, nullptr);
            }));
        }
    }

    return results;
}

}  // namespace bench
