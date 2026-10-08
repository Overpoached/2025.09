// Benchmark runner for the CUDA inference engine.
//
// Everything this project claims about performance flows through this binary.
// It exists before any application kernel does, by design: a measurement you
// cannot reproduce is not a measurement.

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bench/bandwidth_suite.h"
#include "bench/cuda_check.h"
#include "bench/gpu_state.h"
#include "bench/report.h"
#include "bench/stats.h"
#include "kernel.h"

#include "engine/engine.h"
#include "test/test.h"

namespace {

struct Options {
    std::string command = "bench";
    int device_index = 0;
    int warmup_iters = 25;
    int measure_iters = 200;
    int repeats = 1;
    std::vector<int> element_counts;
    std::string markdown_path;
    std::string what_changed;

	// Engine use options
	std::string model_path = "./model/model.onnx";  // Default model path
    std::string image_path = "./images/test.jpg";  // Default image path
};

void PrintUsage(std::ostream& out) {
    out << "usage: cuda [command] [options]\n"
           "\n"
           "commands:\n"
           "  bench                 run the bandwidth benchmark suite (default)\n"
           "  info                  print device properties and current clock state\n"
           "  demo                  run the original add-one/reduction sanity demo\n"
           "\n"
           "options:\n"
           "  --device N            CUDA device index (default 0)\n"
           "  --warmup N            untimed warmup iterations (default 25)\n"
           "  --iters N             timed iterations per configuration (default 200)\n"
           "  --repeats N           repeat the whole suite N times and report the\n"
           "                        run-to-run spread (default 1)\n"
           "  --sizes LIST          comma-separated element counts, K/M suffixes\n"
           "                        allowed, e.g. 1M,4M,16M,64M\n"
           "  --markdown PATH       append a dated BENCHMARKS.md entry to PATH\n"
           "  --note TEXT           the 'what changed' line for that entry\n"
           "  -h, --help            this message\n";
}

// Accepts a plain integer or one with a Ki/Mi suffix: 1024, 64K, 16M.
int ParseElementCount(const std::string& token) {
    if (token.empty()) {
        throw std::runtime_error("empty element count");
    }
    std::size_t consumed = 0;
    const long long base = std::stoll(token, &consumed);
    long long multiplier = 1;
    if (consumed < token.size()) {
        const char suffix = token[consumed];
        if (suffix == 'K' || suffix == 'k') {
            multiplier = 1 << 10;
        } else if (suffix == 'M' || suffix == 'm') {
            multiplier = 1 << 20;
        } else {
            throw std::runtime_error("unrecognised size suffix in '" + token + "'");
        }
        ++consumed;
    }
    if (consumed != token.size()) {
        throw std::runtime_error("trailing characters in size '" + token + "'");
    }
    const long long value = base * multiplier;
    if (value <= 0 || value > 0x7fffffffLL) {
        throw std::runtime_error("element count out of range: " + token);
    }
    return static_cast<int>(value);
}

std::vector<int> ParseSizeList(const std::string& list) {
    std::vector<int> counts;
    std::istringstream stream(list);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            counts.push_back(ParseElementCount(token));
        }
    }
    if (counts.empty()) {
        throw std::runtime_error("--sizes given but no sizes parsed");
    }
    return counts;
}

// Returns the value of a flag that requires an argument, advancing the index.
std::string TakeValue(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string(flag) + " requires a value");
    }
    return argv[++i];
}

int ParsePositive(const std::string& text, const char* flag) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::runtime_error(std::string(flag) + " must be positive");
    }
    return value;
}

Options ParseArgs(int argc, char** argv, bool* should_exit) {
    Options options;
    *should_exit = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsage(std::cout);
            *should_exit = true;
            return options;
        } else if (arg == "--device") {
            options.device_index = std::stoi(TakeValue(argc, argv, i, "--device"));
        } else if (arg == "--warmup") {
            options.warmup_iters = ParsePositive(TakeValue(argc, argv, i, "--warmup"), "--warmup");
        } else if (arg == "--iters") {
            options.measure_iters = ParsePositive(TakeValue(argc, argv, i, "--iters"), "--iters");
        } else if (arg == "--repeats") {
            options.repeats = ParsePositive(TakeValue(argc, argv, i, "--repeats"), "--repeats");
        } else if (arg == "--sizes") {
            options.element_counts = ParseSizeList(TakeValue(argc, argv, i, "--sizes"));
        } else if (arg == "--markdown") {
            options.markdown_path = TakeValue(argc, argv, i, "--markdown");
        } else if (arg == "--note") {
            options.what_changed = TakeValue(argc, argv, i, "--note");
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unrecognised option: " + arg);
        } else {
            options.command = arg;
        }
    }

    if (options.element_counts.empty()) {
        options.element_counts = bench::DefaultElementCounts();
    }
    return options;
}

// Across repeated runs of the whole suite, how far apart are the medians? This
// is the week-1 trust check: within-run variance can look tidy while the
// machine still drifts between runs.
void PrintRunToRunSummary(std::ostream& out,
                          const std::vector<std::vector<bench::BenchResult>>& runs) {
    if (runs.size() < 2) {
        return;
    }

    out << "\nRun-to-run spread over " << runs.size() << " repeats of the suite:\n\n"
        << std::left << std::setw(14) << "variant" << std::setw(16) << "configuration"
        << std::right << std::setw(14) << "median ms" << std::setw(14) << "slowest ms"
        << std::setw(11) << "spread %" << "\n"
        << std::string(69, '-') << "\n";

    for (std::size_t config = 0; config < runs.front().size(); ++config) {
        std::vector<double> medians;
        medians.reserve(runs.size());
        for (const std::vector<bench::BenchResult>& run : runs) {
            medians.push_back(run[config].time_ms.median);
        }
        const bench::Stats across = bench::ComputeStats(medians);
        const double spread = across.min > 0.0 ? 100.0 * (across.max - across.min) / across.min : 0.0;

        out << std::left << std::setw(14) << runs.front()[config].config.variant << std::setw(16)
            << runs.front()[config].config.configuration << std::right << std::fixed
            << std::setprecision(4) << std::setw(14) << across.median << std::setw(14) << across.max
            << std::setprecision(2) << std::setw(11) << spread << "\n";
    }
}

int RunInfo(const Options& options) {
    const bench::DeviceInfo device = bench::QueryDeviceInfo(options.device_index);
    const bench::GpuSample state = bench::SampleGpuState(options.device_index);
    bench::PrintDeviceBanner(std::cout, device, state);
    return 0;
}

int RunDemo() {
    constexpr int kSize = 1000;
    std::vector<int> data(kSize);
    for (int i = 0; i < kSize; ++i) {
        data[i] = i;
    }

    const int expected = (kSize - 1) * kSize / 2 + kSize;
    const int sum = addOneAndSum(data.data(), kSize);

    std::cout << "sum after adding one to each element: " << sum << " (expected " << expected
              << ")\n";
    if (sum != expected) {
        std::cout << "FAILED\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

int RunBench(const Options& options) {
    const bench::DeviceInfo device = bench::QueryDeviceInfo(options.device_index);
    const bench::GpuSample state_at_start = bench::SampleGpuState(options.device_index);
    bench::PrintDeviceBanner(std::cout, device, state_at_start);

    bench::BandwidthSuiteOptions suite;
    suite.element_counts = options.element_counts;
    suite.warmup_iters = options.warmup_iters;
    suite.measure_iters = options.measure_iters;
    suite.device_index = options.device_index;

    std::vector<std::vector<bench::BenchResult>> runs;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        if (options.repeats > 1) {
            std::cout << "\n--- suite repeat " << (repeat + 1) << " of " << options.repeats
                      << " ---\n";
        } else {
            std::cout << "\n";
        }
        runs.push_back(bench::RunBandwidthSuite(suite, std::cout));
        bench::PrintResultsTable(std::cout, runs.back(), device);
    }

    PrintRunToRunSummary(std::cout, runs);

    const std::vector<bench::BenchResult>& final_run = runs.back();
    const bool trustworthy = bench::PrintMeasurementWarnings(std::cout, final_run);

    if (!options.markdown_path.empty()) {
        const bench::GpuSample state_at_end = bench::SampleGpuState(options.device_index);
        const std::string entry = bench::MakeMarkdownEntry(
            final_run, device, state_at_end, "vector_add bandwidth baseline",
            options.what_changed.empty() ? "(no note given)" : options.what_changed);

        std::ofstream file(options.markdown_path, std::ios::app);
        if (!file) {
            std::cerr << "could not open " << options.markdown_path << " for appending\n";
            return 1;
        }
        file << "\n" << entry;
        std::cout << "\nAppended an entry to " << options.markdown_path << "\n";
        if (!trustworthy) {
            std::cout << "Note the warnings above before trusting that entry.\n";
        }
    }

    return trustworthy ? 0 : 2;
}

int RunEngine(const Options& options) {
    ohy::Engine engine;
    engine.Initialize(options.model_path);  // Default path
    std::string result = engine.ProcessImage(options.image_path);
    std::cout << "Identified car model: " << result << "\n";

	engine.Shutdown();
    return 0;
}

int RunTest(const Options& options) {
    std::cout << "Running Test...\n";
	test::InitTests();
    return test::RunAllTests();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool should_exit = false;
        const Options options = ParseArgs(argc, argv, &should_exit);
        if (should_exit) {
            return 0;
        }

        CUDA_CHECK(cudaSetDevice(options.device_index));

        if (options.command == "info") {
            return RunInfo(options);
        }
        if (options.command == "demo") {
            return RunDemo();
        }
        if (options.command == "bench") {
            return RunBench(options);
        }
        if (options.command == "add") {
			return RunEngine(options);
        }
        if (options.command == "test") {
            return RunTest(options);
        }

        std::cerr << "unknown command: " << options.command << "\n\n";
        PrintUsage(std::cerr);
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "\nerror: " << error.what() << "\n";
        return 1;
    }
}
