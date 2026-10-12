#include "test.h"

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <sstream>

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
		ohy::Buffer d_x(ohy::Location::Device, reinterpret_cast<const char*>(x.data()), bytes);
		ohy::Buffer d_y(ohy::Location::Device, bytes);

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
		ohy::Buffer d_x(ohy::Location::Device, reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
		ohy::Buffer d_w(ohy::Location::Device, reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
		ohy::Buffer d_p(ohy::Location::Device, reinterpret_cast<const char*>(pad.data()), pad.size() * sizeof(int64_t));
		ohy::Buffer d_s(ohy::Location::Device, reinterpret_cast<const char*>(stride.data()), stride.size() * sizeof(int64_t));
		ohy::Buffer d_y(ohy::Location::Device, ybytes);

		kernels::ConvShape shapes;
		shapes.xbatch = shapes.ybatch = 1;
		shapes.xchannels = 1;
		shapes.xrows = xrows;
		shapes.xcols = xcols;
		shapes.wcount = 1;
		shapes.wrows = wrows;
		shapes.wcols = wcols;
		shapes.ychannels = shapes.wcount;
		shapes.yrows = yh;
		shapes.ycols = yw;
		shapes.uppad = pad[0];
		shapes.leftpad = pad[1];
		shapes.downpad = pad[2];
		shapes.rightpad = pad[3];
		shapes.rowstride = stride[0];
		shapes.colstride = stride[1];
		kernels::Conv(static_cast<const float*>(d_x.ptr),
			static_cast<float*>(d_w.ptr),
			static_cast<float*>(d_y.ptr), shapes, 0);
		CUDA_CHECK(cudaDeviceSynchronize());           // surfaces errors from inside the kernel

		std::vector<float> y(shapes.xchannels * shapes.wcount * yw * yh);
		CUDA_CHECK(cudaMemcpy(y.data(), d_y.ptr, ybytes, cudaMemcpyDeviceToHost));

		if (print_result) {
			std::cout << "y:\n" << ohy::ToString(y) << std::endl;
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

	// ---------------------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------------------
//
// Layouts (all row-major, contiguous):
//   A    [batch, M, K]                        batch stride M*K
//   B    [batch, K, N]   (transB = false)     batch stride K*N, or 0 when shareB
//        [batch, N, K]   (transB = true)      i.e. B[k][n] lives at n*K + k  (ONNX Gemm transB=1)
//   bias [N] or empty                         added to every row of C
//   C    [batch, M, N]                        batch stride M*N

	struct MatMulCase {
		int64_t batch = 1, M = 0, N = 0, K = 0;
		bool transB = false;
		bool shareB = false;             // one B for all batches (ONNX: N-D activation x 2-D weight)
		std::vector<float> a, b, bias;
		std::vector<float> expected;
		std::vector<double> tolerance;   // per element; empty -> 1e-6 (hand-written cases are exact)
	};

	// The only function that knows about your kernel. Adapt this call to your interface.
	void LaunchMatMul(const kernels::GEMMShape& c, const float* dA, const float* dB, const float* dBias, float* dC) {

		kernels::GEMM(dA, dB, dBias, dC, c, 0);
	}

	void RunMatMulCase(const MatMulCase& c, bool print_result = false) {
		const size_t aCount = static_cast<size_t>(c.batch * c.M * c.K);
		const size_t bCount = static_cast<size_t>((c.shareB ? 1 : c.batch) * c.K * c.N);
		const size_t cCount = static_cast<size_t>(c.batch * c.M * c.N);
		if (c.a.size() != aCount || c.b.size() != bCount || c.expected.size() != cCount
			|| (!c.bias.empty() && c.bias.size() != static_cast<size_t>(c.N))) {
			throw std::runtime_error("MatMulCase: data sizes do not match batch/M/N/K");   // bug in the test, not the kernel
		}

		ohy::Buffer dA(ohy::Location::Device, reinterpret_cast<const char*>(c.a.data()), aCount * sizeof(float));
		ohy::Buffer dB(ohy::Location::Device, reinterpret_cast<const char*>(c.b.data()), bCount * sizeof(float));
		std::unique_ptr<ohy::Buffer> dBias;
		if (!c.bias.empty()) {
			dBias = std::make_unique<ohy::Buffer>(ohy::Location::Device, reinterpret_cast<const char*>(c.bias.data()), c.bias.size() * sizeof(float));
		}


		std::vector<int64_t> ashape{ c.batch, c.M, c.K };
		std::vector<int64_t> bshape{ c.shareB ? 1 : c.batch, c.K, c.N };
		std::vector<int64_t> outShape;
		kernels::GEMMShape shape = kernels::MakeMatMulShape(ashape, bshape, outShape);
		shape.transB = c.transB;
		shape.hasC = c.bias.empty() ? 0 : 1;

		ohy::Buffer dC(ohy::Location::Device, cCount * sizeof(float));
		CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, cCount * sizeof(float)));   // NaN fill: any element the kernel skips fails

		LaunchMatMul(shape, static_cast<const float*>(dA.ptr), static_cast<const float*>(dB.ptr),
			dBias ? static_cast<const float*>(dBias->ptr) : nullptr, static_cast<float*>(dC.ptr));
		CUDA_CHECK(cudaDeviceSynchronize());           // surfaces errors from inside the kernel

		std::vector<float> got(cCount);
		CUDA_CHECK(cudaMemcpy(got.data(), dC.ptr, cCount * sizeof(float), cudaMemcpyDeviceToHost));

		if (print_result) {
			std::cout << "result: " << c.batch << " batches\n";
			for (int b = 0; b < c.batch; ++b) {
				std::cout << b << ":\n[ ";
				for (int j = 0; j < c.M; ++j) {
					if (j > 0) std::cout << "\n";
					for (int i = 0; i < c.N; ++i) {
						if (i > 0) std::cout << ", ";
						std::cout << got[b * c.M * c.N + j * c.N + i];
					}
				}
				std::cout << " ]\n";
			}
				
		}

		for (size_t i = 0; i < cCount; ++i) {
			const double tol = c.tolerance.empty() ? 1e-6 : c.tolerance[i];
			const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(c.expected[i]));
			if (!(diff <= tol)) {   // written this way so NaN fails too
				const int64_t b = static_cast<int64_t>(i) / (c.M * c.N);
				const int64_t r = (static_cast<int64_t>(i) / c.N) % c.M;
				const int64_t col = static_cast<int64_t>(i) % c.N;
				std::ostringstream os;
				os.precision(9);
				os << "[batch " << c.batch << ", M " << c.M << ", N " << c.N << ", K " << c.K
					<< (c.transB ? ", transB" : "") << (c.shareB ? ", shareB" : "") << (c.bias.empty() ? "" : ", bias") << "] "
					<< "C[" << b << "][" << r << "][" << col << "] got " << got[i] << ", expected " << c.expected[i]
					<< " (diff " << diff << " > tol " << tol << ")";
				throw detail::Failure(os.str());
			}
		}
	}

	// Random inputs in [-1, 1), expected computed in double.
	// Tolerance is relative to sum|a*b| per element: fp32 accumulation error scales with it,
	// while an indexing bug is off by O(1) and fails by orders of magnitude.
	MatMulCase MakeRandomMatMulCase(int64_t batch, int64_t M, int64_t N, int64_t K,
		bool transB = false, bool shareB = false, bool withBias = false, uint32_t seed = 1234)
	{
		MatMulCase c;
		c.batch = batch; c.M = M; c.N = N; c.K = K;
		c.transB = transB; c.shareB = shareB;

		std::mt19937 rng(seed);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		c.a.resize(static_cast<size_t>(batch * M * K));
		for (auto& v : c.a) v = dist(rng);
		c.b.resize(static_cast<size_t>((shareB ? 1 : batch) * K * N));
		for (auto& v : c.b) v = dist(rng);
		if (withBias) {
			c.bias.resize(static_cast<size_t>(N));
			for (auto& v : c.bias) v = dist(rng);
		}

		c.expected.resize(static_cast<size_t>(batch * M * N));
		c.tolerance.resize(c.expected.size());
		for (int64_t b = 0; b < batch; ++b) {
			const float* A = c.a.data() + b * M * K;
			const float* B = c.b.data() + (shareB ? 0 : b * K * N);
			for (int64_t r = 0; r < M; ++r) {
				for (int64_t col = 0; col < N; ++col) {
					double sum = withBias ? c.bias[col] : 0.0;
					double sumAbs = std::fabs(sum);
					for (int64_t k = 0; k < K; ++k) {
						const double bv = transB ? B[col * K + k] : B[k * N + col];
						const double p = static_cast<double>(A[r * K + k]) * bv;
						sum += p;
						sumAbs += std::fabs(p);
					}
					const size_t i = static_cast<size_t>((b * M + r) * N + col);
					c.expected[i] = static_cast<float>(sum);
					c.tolerance[i] = 1e-5 * sumAbs + 1e-6;
				}
			}
		}
		return c;
	}

	// ---- Hand-written: small integer matrices, exact results ---------------------------

	TEST_CASE(MatMulSquare2x2) {
		MatMulCase c;
		c.M = 2; c.N = 2; c.K = 2;
		c.a = { 1, 2,
				3, 4 };
		c.b = { 5, 6,
				7, 8 };
		c.expected = { 19, 22,
					   43, 50 };
		RunMatMulCase(c);
	}

	// Non-square with M, N, K all different: swapping any two dimensions in the indexing fails this.
	TEST_CASE(MatMulNonSquare2x3x4) {
		MatMulCase c;
		c.M = 2; c.N = 4; c.K = 3;
		c.a = { 1, 2, 3,
				4, 5, 6 };
		c.b = { 1,  2,  3,  4,
				5,  6,  7,  8,
				9, 10, 11, 12 };
		c.expected = { 38, 44,  50,  56,
					   83, 98, 113, 128 };
		RunMatMulCase(c);
	}

	// M = 1: a single row, like the final [4,768] x [768,512] projection when batch is 1.
	TEST_CASE(MatMulRowVector) {
		MatMulCase c;
		c.M = 1; c.N = 2; c.K = 3;
		c.a = { 1, 2, 3 };
		c.b = { 1, 2,
				3, 4,
				5, 6 };
		c.expected = { 22, 28 };
		RunMatMulCase(c);
	}

	// K = 1: outer product. Catches loops that assume K is at least one tile.
	TEST_CASE(MatMulOuterProductK1) {
		MatMulCase c;
		c.M = 3; c.N = 2; c.K = 1;
		c.a = { 1,
				2,
				3 };
		c.b = { 4, 5 };
		c.expected = { 4,  5,
						8, 10,
					   12, 15 };
		RunMatMulCase(c);
	}

	// Two independent multiplies with different A AND different B (attention: Q x K^T per head).
	// A batch-stride bug typically gives the right answer for batch 0 and wrong for batch 1.
	TEST_CASE(MatMulBatched) {
		MatMulCase c;
		c.batch = 2; c.M = 2; c.N = 2; c.K = 2;
		c.a = { 1, 2,   3, 4,      // batch 0
				2, 0,   1, 1 };    // batch 1
		c.b = { 5, 6,   7, 8,      // batch 0
				1, 2,   3, 4 };    // batch 1
		c.expected = { 19, 22,   43, 50,
						2,  4,    4,  6 };
		RunMatMulCase(c);
	}

	// Same B for every batch item (activation [2,2,3] x weight [3,2]). Must equal the flattened M=4 case below.
	TEST_CASE(MatMulSharedB) {
		MatMulCase c;
		c.batch = 2; c.M = 2; c.N = 2; c.K = 3;
		c.shareB = true;
		c.a = { 1,  2,  3,    4,  5,  6,      // batch 0
				 7,  8,  9,   10, 11, 12 };    // batch 1
		c.b = { 1, 2,
				3, 4,
				5, 6 };
		c.expected = { 22,  28,   49,  64,
					   76, 100,  103, 136 };
		RunMatMulCase(c);
	}

	// Same data as MatMulSharedB with the leading dims collapsed into M: the "flatten" path.
	TEST_CASE(MatMulFlattenedLeadingDims) {
		MatMulCase c;
		c.M = 4; c.N = 2; c.K = 3;
		c.a = { 1,  2,  3,
				 4,  5,  6,
				 7,  8,  9,
				10, 11, 12 };
		c.b = { 1, 2,
				3, 4,
				5, 6 };
		c.expected = { 22,  28,
					   49,  64,
					   76, 100,
					  103, 136 };
		RunMatMulCase(c);
	}

	// Gemm transB=1: same maths as MatMulNonSquare2x3x4, but B is stored as [N, K].
	TEST_CASE(MatMulTransB) {
		MatMulCase c;
		c.M = 2; c.N = 4; c.K = 3;
		c.transB = true;
		c.a = { 1, 2, 3,
				4, 5, 6 };
		c.b = { 1, 5,  9,      // column 0 of the original B
				2, 6, 10,	   // { 1, 2, 3, 4
				3, 7, 11,	   //   5, 6, 7, 8
				4, 8, 12 };    //   9, 10, 11, 12 };
		c.expected = { 38, 44,  50,  56,
					   83, 98, 113, 128 };
		RunMatMulCase(c);
	}

	// Gemm as CLIP uses it: transB=1 plus a per-column bias. A per-row bias (wrong) gives different numbers.
	TEST_CASE(MatMulTransBBias) {
		MatMulCase c;
		c.M = 2; c.N = 4; c.K = 3;
		c.transB = true;
		c.a = { 1, 2, 3,
				4, 5, 6 };
		c.b = { 1, 5,  9,
				2, 6, 10,
				3, 7, 11,
				4, 8, 12 };
		c.bias = { 1, -2, 10, 0.5f };
		c.expected = { 39, 42,  60,  56.5f,
					   84, 96, 123, 128.5f };
		RunMatMulCase(c, true);
	}

	// ---- Generated: awkward sizes (tile edges) ------------------------------------------

	TEST_CASE(MatMulRandomEdgeSizes) {
		const int64_t sizes[][3] = {   // M, N, K
			{   1,   1,   1 },
			{  17,  19,  23 },         // all smaller than any tile
			{  31,  33,  65 },         // one either side of 32, just over 64
			{  64,  64,  64 },         // exact multiple: should pass even if edges are broken
			{  65, 127, 129 },         // one past a tile boundary in every dimension
			{ 197, 131,  73 },         // primes
		};
		for (const auto& s : sizes) {
			RunMatMulCase(MakeRandomMatMulCase(1, s[0], s[1], s[2]));
			RunMatMulCase(MakeRandomMatMulCase(1, s[0], s[1], s[2], /*transB*/ true, false, /*bias*/ true));
			RunMatMulCase(MakeRandomMatMulCase(3, s[0], s[1], s[2]));                   // batched
			RunMatMulCase(MakeRandomMatMulCase(3, s[0], s[1], s[2], false, /*shareB*/ true));
		}
	}

	// ---- Generated: the exact shapes CLIP ViT-B/32 runs at batch 4 -----------------------
	// The CPU reference is slow in Debug (a few seconds each); run these in Release if it bothers you.

	TEST_CASE(MatMulClipQkvProjection) { RunMatMulCase(MakeRandomMatMulCase(1, 200, 2304, 768)); }
	TEST_CASE(MatMulClipOutProjection) { RunMatMulCase(MakeRandomMatMulCase(1, 200, 768, 768, true, false, true)); }
	TEST_CASE(MatMulClipMlpExpand) { RunMatMulCase(MakeRandomMatMulCase(1, 200, 3072, 768)); }
	TEST_CASE(MatMulClipMlpContract) { RunMatMulCase(MakeRandomMatMulCase(1, 200, 768, 3072)); }
	TEST_CASE(MatMulClipAttentionScores) { RunMatMulCase(MakeRandomMatMulCase(48, 50, 50, 64)); }
	TEST_CASE(MatMulClipAttentionValues) { RunMatMulCase(MakeRandomMatMulCase(48, 50, 64, 50)); }
	TEST_CASE(MatMulClipFinalProjection) { RunMatMulCase(MakeRandomMatMulCase(1, 4, 512, 768)); }

	void LaunchGather(const std::vector<float>& data,
		const std::vector<int64_t>& datashape,
		const std::vector<int64_t>& indices,
		const std::vector<int64_t>& indicesshape,
		int32_t axis,
		const std::vector<float>& output,
		bool print_result = false)
	{
		const int64_t rank = static_cast<int64_t>(datashape.size());
		if (axis < 0) {
			axis += static_cast<int32_t>(rank);
		}
		if (axis < 0 || axis >= rank) {
			throw std::runtime_error("LaunchGather: axis out of range");   // bug in the test, not the kernel
		}

		// data viewed as [outer, axisDim, inner]
		kernels::GatherShape shape{};
		shape.outer = 1;
		shape.inner = 1;
		for (int64_t d = 0; d < axis; ++d) { shape.outer *= datashape[d]; }
		for (int64_t d = axis + 1; d < rank; ++d) { shape.inner *= datashape[d]; }
		shape.axisDim = datashape[axis];

		size_t dataSize = 1, indicesSize = 1;
		for (int64_t v : datashape) { dataSize *= static_cast<size_t>(v); }
		for (int64_t v : indicesshape) { indicesSize *= static_cast<size_t>(v); }   // 1 for a scalar index ({})
		shape.numIndices = static_cast<int64_t>(indicesSize);
		shape.n = shape.outer * shape.numIndices * shape.inner;

		// data.shape[:axis] + indices.shape + data.shape[axis+1:]
		std::vector<int64_t> outShape = datashape;
		auto pos = outShape.erase(outShape.begin() + axis);
		outShape.insert(pos, indicesshape.begin(), indicesshape.end());

		if (data.size() != dataSize || indices.size() != indicesSize || output.size() != static_cast<size_t>(shape.n)) {
			throw std::runtime_error("LaunchGather: data/indices/output sizes do not match their shapes");
		}

		ohy::Buffer dA(ohy::Location::Device, reinterpret_cast<const char*>(data.data()), dataSize * sizeof(float));
		ohy::Buffer dB(ohy::Location::Device, reinterpret_cast<const char*>(indices.data()), indicesSize * sizeof(int64_t));
		ohy::Buffer dC(ohy::Location::Device, static_cast<size_t>(shape.n) * sizeof(float));   // (loc, n): allocate only
		CUDA_CHECK(cudaMemset(dC.ptr, 0xFF, static_cast<size_t>(shape.n) * sizeof(float)));       // NaN fill: unwritten elements fail

		kernels::Gather(static_cast<const float*>(dA.ptr), static_cast<const int64_t*>(dB.ptr),
			static_cast<float*>(dC.ptr), shape, 0);
		CUDA_CHECK(cudaDeviceSynchronize());           // surfaces errors from inside the kernel

		std::vector<float> got(static_cast<size_t>(shape.n));
		CUDA_CHECK(cudaMemcpy(got.data(), dC.ptr, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

		if (print_result) {
			std::cout << "out shape " << ohy::ToString(outShape) << "\n" << ohy::ToString(got) << std::endl;
		}

		for (size_t i = 0; i < got.size(); ++i) {
			EXPECT_NEAR(got[i], output[i], 1e-6);   // pure copy: should be exact
		}
	}

	TEST_CASE(Gather) {
		std::vector<int64_t> datashape{ 3, 2 };
		std::vector<int64_t> indicesshape{ 2, 2 };
		//std::vector<int64_t> outputshape{ 2, 2, 2 };

		std::vector<float> data{
			1.0, 1.2,
			2.3, 3.4,
			4.5, 5.7,
		};
		std::vector<int64_t>indices{
			0, 1,
			1, 2,
		};
		int32_t axis = 0;
		std::vector<float> output {
				1.0, 1.2,
				2.3, 3.4,
				2.3, 3.4,
				4.5, 5.7,
		};

		LaunchGather(data, datashape, indices, indicesshape, axis, output);
	}

	TEST_CASE(GatherAxis) {
		std::vector<int64_t> datashape{ 3, 3 };
		std::vector<int64_t> indicesshape{ 1, 2 };
		//std::vector<int64_t> outputshape{ 3, 1, 2 };

		std::vector<float> data{
			1.0, 1.2, 1.9,
			2.3, 3.4, 3.9,
			4.5, 5.7, 5.9
		};
		std::vector<int64_t>indices{
			0, 2,
		};
		int32_t axis = 1;
		std::vector<float> output{
			1.0, 1.9,
			2.3, 3.9,
			4.5, 5.9,
		};

		LaunchGather(data, datashape, indices, indicesshape, axis, output);
	}
	// Scalar index (indices shape {}): the gathered axis disappears. This is CLIP's QKV split and CLS select.
	TEST_CASE(GatherScalarIndex) {
		std::vector<int64_t> datashape{ 3, 2 };
		std::vector<int64_t> indicesshape{};          // scalar
		std::vector<float> data{
			1.0, 1.2,
			2.3, 3.4,
			4.5, 5.7,
		};
		std::vector<int64_t> indices{ 1 };
		std::vector<float> output{ 2.3, 3.4 };        // shape [2]
		LaunchGather(data, datashape, indices, indicesshape, 0, output);
	}

	// Negative index counts from the end: -1 on axis 1 of a [3,3] is column 2. Same result as GatherAxis.
	TEST_CASE(GatherNegativeIndex) {
		std::vector<int64_t> datashape{ 3, 3 };
		std::vector<int64_t> indicesshape{ 1, 2 };
		std::vector<float> data{
			1.0, 1.2, 1.9,
			2.3, 3.4, 3.9,
			4.5, 5.7, 5.9
		};
		std::vector<int64_t> indices{ 0, -1 };
		std::vector<float> output{
			1.0, 1.9,
			2.3, 3.9,
			4.5, 5.9,
		};
		LaunchGather(data, datashape, indices, indicesshape, 1, output);
	}
}
