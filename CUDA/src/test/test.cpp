#include "test.h"

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../engine/tensor.h"
#include "../engine/utility.h"
#include "../kernels/kernel_utility.h"
#include "../kernels/nn/nn.h"

namespace test {

	std::map<std::string, std::function<void()>> testCases;

	namespace detail {
		// Function-local static so the head is valid no matter which translation unit's
		// registrars run first (avoids the static-initialisation-order problem).
		static Registrar*& PendingHead() {
			static Registrar* head = nullptr;
			return head;
		}

		Registrar::Registrar(const char* name, void (*fn)()) : name(name), fn(fn), next(PendingHead()) {
			PendingHead() = this;
		}
	}

	void InitTests() {
		for (detail::Registrar* r = detail::PendingHead(); r != nullptr; r = r->next) {
			testCases[r->name] = r->fn;
		}
	}

	int RunAllTests() {
		InitTests();

		struct Result { std::string name; std::string error; };
		std::vector<Result> failures;
		int passed = 0;

		for (const auto& [name, fn] : testCases) {
			std::cout << "[ RUN  ] " << name << "\n";
			std::string error;
			try {
				fn();
			} catch (const detail::Failure& f) {
				error = f.what();
			} catch (const std::exception& e) {        // e.g. CUDA_CHECK throwing
				error = std::string("exception: ") + e.what();
			} catch (...) {
				error = "unknown exception";
			}

			if (error.empty()) {
				++passed;
				std::cout << "[  OK  ] " << name << "\n";
			} else {
				std::cout << "[ FAIL ] " << name << "\n";
				failures.push_back({ name, error });
			}
		}

		std::cout << "\n" << testCases.size() << " tests: " << passed << " passed, "
		          << failures.size() << " failed\n";
		for (const Result& f : failures) {
			std::cout << "  " << f.name << ": " << f.error << "\n";
		}
		return static_cast<int>(failures.size());
	}

	// ---------------------------------------------------------------------------------------
	// Test cases
	// ---------------------------------------------------------------------------------------

	TEST_CASE(KernelSigmoid) {
		// Not a multiple of the 256-thread block, so the tail guard (i < n) is exercised.
		constexpr int64_t n = 1000 + 3;
		std::vector<float> x(n);
		for (int64_t i = 0; i < n; ++i) {
			x[i] = -20.0f + 40.0f * float(i) / float(n - 1);   // spans [-20, 20]
		}
		x[0] = -100.0f;   // expf(100) overflows to inf -> result must be exactly 0, not NaN
		x[1] = 100.0f;
		x[2] = 0.0f;

		const size_t bytes = n * sizeof(float);
		ohy::DeviceBuffer d_x(reinterpret_cast<const char*>(x.data()), bytes);
		ohy::DeviceBuffer d_y(bytes);

		kernels::Sigmoid(static_cast<const float*>(d_x.ptr), static_cast<float*>(d_y.ptr), n, 0);
		CUDA_CHECK(cudaDeviceSynchronize());           // surfaces errors from inside the kernel

		std::vector<float> y(n);
		CUDA_CHECK(cudaMemcpy(y.data(), d_y.ptr, bytes, cudaMemcpyDeviceToHost));

		EXPECT_EQ(y[0], 0.0f);
		EXPECT_EQ(y[1], 1.0f);
		EXPECT_EQ(y[2], 0.5f);

		// Reference in double. GPU expf is within 2 ulp, so 1e-6 absolute is comfortable
		// for outputs in [0, 1].
		for (int64_t i = 0; i < n; ++i) {
			const double want = 1.0 / (1.0 + std::exp(-double(x[i])));
			EXPECT_NEAR(y[i], want, 1e-6);
		}
	}


	void TestConv(const std::vector<float>& x, int xrows, int xcols,
		const std::vector<float>& w, int wrows, int wcols,
		const std::vector<int64_t>& pad,
		const std::vector<int64_t>& stride,
		const std::vector<float>& r,
		bool print_result = false)
	{
		// out = (in + padBegin + padEnd - K) / stride + 1
		const int64_t yw = (xcols + pad[1] + pad[3] - wcols) / stride[1] + 1;
		const int64_t yh = (xrows + pad[0] + pad[2] - wrows) / stride[0] + 1;
		const size_t ybytes = yw * yh * sizeof(float);
		ohy::DeviceBuffer d_x(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
		ohy::DeviceBuffer d_w(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
		ohy::DeviceBuffer d_p(reinterpret_cast<const char*>(pad.data()), pad.size() * sizeof(int64_t));
		ohy::DeviceBuffer d_s(reinterpret_cast<const char*>(stride.data()), stride.size() * sizeof(int64_t));
		ohy::DeviceBuffer d_y(ybytes);

		kernels::ConvShape shapes;
		shapes.xbatch = shapes.ybatch = 1;
		shapes.xchannels = 1;
		shapes.xrows = xrows;
		shapes.xcols = xcols;
		shapes.wcount = 1;
		shapes.wrows = wrows;
		shapes.wcols = wcols;
		shapes.yrows = yh;
		shapes.ycols = yw;
		kernels::Conv(static_cast<const float*>(d_x.ptr),
			static_cast<float*>(d_w.ptr),
			static_cast<float*>(d_y.ptr), shapes,
			static_cast<int64_t*>(d_p.ptr), pad.size(),
			static_cast<int64_t*>(d_s.ptr), stride.size(), 0);
		CUDA_CHECK(cudaDeviceSynchronize());           // surfaces errors from inside the kernel

		std::vector<float> y(yw * yh);
		CUDA_CHECK(cudaMemcpy(y.data(), d_y.ptr, ybytes, cudaMemcpyDeviceToHost));

		if (print_result) {
			std::cout << "y:\n" << ohy::utility::ToString(y) << std::endl;
		}

		for (int64_t i = 0; i < r.size(); ++i) {
			EXPECT_NEAR(y[i], r[i], 1e-6);
		}
	}

	TEST_CASE(KernelConv) {
		constexpr int xcols = 5;
		constexpr int xrows = 5;
		constexpr int wcols = 3;
		constexpr int wrows = 3;

		std::vector<float> x{
			0.0, 1.0, 2.0, 3.0, 4.0,  // (1, 1, 5, 5) input tensor
			5.0, 6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0, 14.0,
			15.0, 16.0, 17.0, 18.0, 19.0,
			20.0, 21.0, 22.0, 23.0, 24.0,
		};
		std::vector<float> w{
			1.0, 1.0, 1.0,  // (1, 1, 3, 3) kernel tensor
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		};
		std::vector<int64_t> pad{
			1, 1, 1, 1
		};
		std::vector<int64_t> stride{
			1, 1
		};
		std::vector<float> r{
			12.0, 21.0, 27.0, 33.0, 24.0,  // (1, 1, 5, 5) output tensor
			33.0, 54.0, 63.0, 72.0, 51.0,
			63.0, 99.0, 108.0, 117.0, 81.0,
			93.0, 144.0, 153.0, 162.0, 111.0,
			72.0, 111.0, 117.0, 123.0, 84.0,
		};

		TestConv(x, xrows, xcols, w, wrows, wcols, pad, stride, r);
	}

	TEST_CASE(KernelConvNoPadding) {
		constexpr int xcols = 5;
		constexpr int xrows = 5;
		constexpr int wcols = 3;
		constexpr int wrows = 3;

		std::vector<float> x{
			0.0, 1.0, 2.0, 3.0, 4.0,  // (1, 1, 5, 5) input tensor
			5.0, 6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0, 14.0,
			15.0, 16.0, 17.0, 18.0, 19.0,
			20.0, 21.0, 22.0, 23.0, 24.0,
		};
		std::vector<float> w{
			1.0, 1.0, 1.0,  // (1, 1, 3, 3) kernel tensor
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		};
		std::vector<int64_t> pad{
			0, 0, 0, 0
		};
		std::vector<int64_t> stride{
			1, 1
		};
		std::vector<float> r{
			54.0, 63.0, 72.0 ,  // (1, 1, 3, 3) output tensor
			99.0, 108.0, 117.0,
			144.0, 153.0, 162.0,
		};

		TestConv(x, xrows, xcols, w, wrows, wcols, pad, stride, r);
	}

	TEST_CASE(KernelConvStrides) {

		std::vector<float> x{
			0.0, 1.0, 2.0, 3.0, 4.0,  // (1, 1, 7, 5) input tensor
			5.0, 6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0, 14.0,
			15.0, 16.0, 17.0, 18.0, 19.0,
			20.0, 21.0, 22.0, 23.0, 24.0,
			25.0, 26.0, 27.0, 28.0, 29.0,
			30.0, 31.0, 32.0, 33.0, 34.0,
		};
		std::vector<float> w{
			1.0, 1.0, 1.0,  // (1, 1, 3, 3) kernel tensor
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		};
		std::vector<int64_t> pad{
			1, 1, 1, 1
		};
		std::vector<int64_t> stride{
			2, 2
		};
		std::vector<float> r{
			12.0, 27.0, 24.0,  // (1, 1, 4, 3) output tensor
			63.0, 108.0, 81.0,
			123.0, 198.0, 141.0,
			112.0, 177.0, 124.0,
		};

		TestConv(x, 7, 5, w, 3, 3, pad, stride, r);
	}

	TEST_CASE(KernelConvStridesNoPadding) {

		std::vector<float> x{
			0.0, 1.0, 2.0, 3.0, 4.0,  // (1, 1, 7, 5) input tensor
			5.0, 6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0, 14.0,
			15.0, 16.0, 17.0, 18.0, 19.0,
			20.0, 21.0, 22.0, 23.0, 24.0,
			25.0, 26.0, 27.0, 28.0, 29.0,
			30.0, 31.0, 32.0, 33.0, 34.0,
		};
		std::vector<float> w{
			1.0, 1.0, 1.0,  // (1, 1, 3, 3) kernel tensor
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		};
		std::vector<int64_t> pad{
			0, 0, 0, 0
		};
		std::vector<int64_t> stride{
			2, 2
		};
		std::vector<float> r{
			54.0, 72.0,  // (1, 1, 3, 2) output tensor
			144.0, 162.0,
			234.0, 252.0,
		};

		TestConv(x, 7, 5, w, 3, 3, pad, stride, r);
	}

	TEST_CASE(KernelConvStridesAsymPadding) {

		std::vector<float> x{
			0.0, 1.0, 2.0, 3.0, 4.0,  // (1, 1, 7, 5) input tensor
			5.0, 6.0, 7.0, 8.0, 9.0,
			10.0, 11.0, 12.0, 13.0, 14.0,
			15.0, 16.0, 17.0, 18.0, 19.0,
			20.0, 21.0, 22.0, 23.0, 24.0,
			25.0, 26.0, 27.0, 28.0, 29.0,
			30.0, 31.0, 32.0, 33.0, 34.0,
		};
		std::vector<float> w{
			1.0, 1.0, 1.0,  // (1, 1, 3, 3) kernel tensor
			1.0, 1.0, 1.0,
			1.0, 1.0, 1.0,
		};
		std::vector<int64_t> pad{
			1, 0, 1, 0
		};
		std::vector<int64_t> stride{
			2, 2
		};
		std::vector<float> r{
			21.0, 33.0,  // (1, 1, 4, 2) output tensor
			99.0, 117.0,
			189.0, 207.0,
			171.0, 183.0,
		};

		TestConv(x, 7, 5, w, 3, 3, pad, stride, r);
	}
}
