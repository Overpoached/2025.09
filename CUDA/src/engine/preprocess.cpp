#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

#include "tensor.h"
#include "../kernels/kernel_utility.h"
#include "../kernels/preprocess/preprocess.h"

namespace ohy {
	namespace {

		// Pillow's bicubic (a = -0.5), support 2.
		double BicubicFilter(double x) {
			constexpr double a = -0.5;
			x = std::fabs(x);
			if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
			if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
			return 0.0;
		}

		struct AxisTable {
			std::vector<int32_t> packed;   // [first(n) | count(n) | coeffs(n * ksize)]
			int32_t ksize = 0;
			int32_t n = 0;
			int32_t* first() { return packed.data(); }
			int32_t* count() { return packed.data() + n; }
		};

		// Port of Pillow's precompute_coeffs + normalize_coeffs_8bpc (Resample.c),
		// restricted to output pixels [cropStart, cropStart + cropLen).
		// When shrinking, the filter is widened by the shrink factor (that is the antialiasing).
		AxisTable BuildAxis(int inSize, int outSize, int cropStart, int cropLen) {
			const double scale = static_cast<double>(inSize) / outSize;
			const double filterscale = std::max(scale, 1.0);
			const double support = 2.0 * filterscale;

			AxisTable t;
			t.n = cropLen;
			t.ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
			t.packed.assign(static_cast<size_t>(cropLen) * (2 + t.ksize), 0);
			int32_t* coeffs = t.packed.data() + 2 * cropLen;

			std::vector<double> w(t.ksize);
			for (int i = 0; i < cropLen; ++i) {
				const double center = (cropStart + i + 0.5) * scale;
				const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
				const int xmax = std::min(static_cast<int>(center + support + 0.5), inSize) - xmin;

				double ww = 0.0;
				for (int x = 0; x < xmax; ++x) {
					w[x] = BicubicFilter((x + xmin - center + 0.5) / filterscale);
					ww += w[x];
				}
				for (int x = 0; x < xmax; ++x) {
					const double k = (ww != 0.0) ? w[x] / ww : w[x];
					const double f = k * (1 << kernels::kResamplePrecisionBits);
					coeffs[static_cast<size_t>(i) * t.ksize + x] = static_cast<int32_t>(k < 0 ? f - 0.5 : f + 0.5);
				}
				t.first()[i] = xmin;
				t.count()[i] = xmax;
			}
			return t;
		}

		kernels::ResampleAxis DeviceAxis(const Buffer& buf, const AxisTable& t) {
			const int32_t* p = static_cast<const int32_t*>(buf.ptr);
			return { p, p + t.n, p + 2 * t.n, t.ksize, t.n };
		}
	}

	std::vector<uint8_t> ReadFileBytes(const std::string& path) {
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		if (!f) { throw std::runtime_error("cannot open image: " + path); }
		std::vector<uint8_t> bytes(static_cast<size_t>(f.tellg()));
		f.seekg(0);
		f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
		return bytes;
	}

	void PreprocessImage(const std::vector<uint8_t>& encoded, const PreprocessConfig& cfg,
		float* dst, cudaStream_t stream)
	{
		// Decode on CPU. IGNORE_ORIENTATION: PIL's Image.open does not apply EXIF rotation, so neither do we.
		// IMREAD_COLOR: always 3 channels, BGR order (grey is replicated, alpha dropped).
		const cv::Mat raw(1, static_cast<int>(encoded.size()), CV_8U, const_cast<uint8_t*>(encoded.data()));
		cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
		if (bgr.empty()) { throw std::runtime_error("image decode failed"); }
		if (!bgr.isContinuous()) { bgr = bgr.clone(); }

		const int w = bgr.cols, h = bgr.rows, S = cfg.size;

		// torchvision Resize(S): short side -> S, long side = int(S * long / short) (truncated)
		int rw, rh;
		if (w <= h) { rw = S; rh = static_cast<int>(static_cast<double>(S) * h / w); }
		else { rh = S; rw = static_cast<int>(static_cast<double>(S) * w / h); }

		// torchvision CenterCrop(S): int(round((size - S) / 2)), Python round = half to even = nearbyint
		const int top = static_cast<int>(std::nearbyint((rh - S) / 2.0));
		const int left = static_cast<int>(std::nearbyint((rw - S) / 2.0));

		AxisTable hx = BuildAxis(w, rw, left, S);
		AxisTable vy = BuildAxis(h, rh, top, S);

		// Only the source rows the vertical pass will read need a horizontal pass.
		const int rowStart = vy.first()[0];
		const int rowEnd = vy.first()[S - 1] + vy.count()[S - 1];
		for (int i = 0; i < S; ++i) { vy.first()[i] -= rowStart; }   // relative to tmp row 0

		Buffer src(Location::Device, reinterpret_cast<const char*>(bgr.data), static_cast<size_t>(w) * h * 3);
		Buffer tmp(Location::Device, static_cast<size_t>(rowEnd - rowStart) * S * 3);
		Buffer hBuf(Location::Device, reinterpret_cast<const char*>(hx.packed.data()), hx.packed.size() * sizeof(int32_t));
		Buffer vBuf(Location::Device, reinterpret_cast<const char*>(vy.packed.data()), vy.packed.size() * sizeof(int32_t));

		kernels::ImageNormalize norm{};
		std::copy(cfg.mean, cfg.mean + 3, norm.mean);
		std::copy(cfg.std, cfg.std + 3, norm.std);

		kernels::ResizeRows(static_cast<const uint8_t*>(src.ptr), w, rowStart, rowEnd - rowStart,
			static_cast<uint8_t*>(tmp.ptr), DeviceAxis(hBuf, hx), stream);
		kernels::ResizeColsNormalize(static_cast<const uint8_t*>(tmp.ptr), S,
			DeviceAxis(vBuf, vy), norm, dst, stream);

		// Buffers are freed on return; finish the work explicitly rather than relying on cudaFree's implicit sync.
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}
}