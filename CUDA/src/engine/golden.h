// golden.h — test/debug only
#pragma once
#include "../engine/tensor.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ohy {

	class Golden {
	public:
		struct Entry { std::string file; std::string dtype; std::vector<int64_t> shape; };

		Golden(std::string dir, float atol = 1e-4f, float rtol = 1e-3f)
			: dir_(std::move(dir)), atol_(atol), rtol_(rtol)
		{
			std::ifstream f(dir_ + "/manifest.tsv");
			if (!f) { throw std::runtime_error("Golden: cannot open manifest in " + dir_); }
			std::string line;
			while (std::getline(f, line)) {
				std::istringstream ss(line);
				std::string name, file, dtype, shapeStr;
				std::getline(ss, name, '\t'); std::getline(ss, file, '\t');
				std::getline(ss, dtype, '\t'); std::getline(ss, shapeStr, '\t');
				Entry e{ file, dtype, {} };
				std::istringstream sh(shapeStr);
				for (std::string d; std::getline(sh, d, ',');) { if (!d.empty()) e.shape.push_back(std::stoll(d)); }
				entries_[name] = std::move(e);
			}
		}

		std::vector<char> Load(const std::string& name) const {
			const Entry& e = entries_.at(name);
			std::ifstream f(dir_ + "/" + e.file, std::ios::binary | std::ios::ate);
			if (!f) { throw std::runtime_error("Golden: cannot open " + e.file); }
			std::vector<char> buf(static_cast<size_t>(f.tellg()));
			f.seekg(0);
			f.read(buf.data(), buf.size());
			return buf;
		}

		const Entry* Find(const std::string& name) const {
			auto it = entries_.find(name);
			return it == entries_.end() ? nullptr : &it->second;
		}

		// Returns true on pass. Prints one line per tensor.
		bool Check(const std::string& name, const std::string& opType, const Tensor& t, cudaStream_t stream) const {
			const Entry* e = Find(name);
			if (!e) { printf("[SKIP] %-40s %-10s no golden\n", name.c_str(), opType.c_str()); return true; }

			if (t.shape != e->shape) {
				printf("[FAIL] %-40s %-10s shape got %s expected %s\n", name.c_str(), opType.c_str(),
					ToStr(t.shape).c_str(), ToStr(e->shape).c_str());
				return false;
			}
			if (t.dtype != DataType::FLOAT32 || e->dtype != "float32") {
				// int64 tensors are shape metadata: compare exactly
				std::vector<char> ref = Load(name);
				std::vector<char> got = ToHost(t, stream, ref.size());
				bool ok = got == ref;
				printf("[%s] %-40s %-10s int64 exact\n", ok ? "PASS" : "FAIL", name.c_str(), opType.c_str());
				return ok;
			}

			const size_t n = static_cast<size_t>(t.numel());
			std::vector<char> refBytes = Load(name);
			if (refBytes.size() != n * sizeof(float)) { throw std::runtime_error("Golden: size mismatch for " + name); }
			std::vector<char> gotBytes = ToHost(t, stream, n * sizeof(float));
			const float* ref = reinterpret_cast<const float*>(refBytes.data());
			const float* got = reinterpret_cast<const float*>(gotBytes.data());

			int64_t bad = 0, worst = -1;
			double maxAbs = 0.0, maxRel = 0.0;
			for (size_t i = 0; i < n; ++i) {
				if (std::isnan(got[i]) != std::isnan(ref[i])) { ++bad; worst = static_cast<int64_t>(i); continue; }
				double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
				double rel = diff / std::max(std::fabs(static_cast<double>(ref[i])), 1e-6);
				if (diff > maxAbs) { maxAbs = diff; worst = static_cast<int64_t>(i); }
				maxRel = std::max(maxRel, rel);
				if (diff > atol_ + rtol_ * std::fabs(ref[i])) { ++bad; }   // numpy.allclose rule
			}

			bool ok = bad == 0;
			printf("[%s] %-40s %-10s %-18s maxAbs %.3e maxRel %.3e bad %lld/%zu",
				ok ? "PASS" : "FAIL", name.c_str(), opType.c_str(), ToStr(t.shape).c_str(),
				maxAbs, maxRel, static_cast<long long>(bad), n);
			if (worst >= 0) { printf("  worst[%lld] got %.6f ref %.6f", static_cast<long long>(worst), got[worst], ref[worst]); }
			printf("\n");
			return ok;
		}

	private:
		static std::vector<char> ToHost(const Tensor& t, cudaStream_t stream, size_t bytes) {
			std::vector<char> h(bytes);
			if (t.storage->location == Location::Device) {
				cudaStreamSynchronize(stream);   // the kernel must have finished before we read
				cudaMemcpy(h.data(), t.data(), bytes, cudaMemcpyDeviceToHost);
			}
			else {
				std::memcpy(h.data(), t.data(), bytes);
			}
			return h;
		}

		static std::string ToStr(const std::vector<int64_t>& s) {
			std::string r = "[";
			for (size_t i = 0; i < s.size(); ++i) { r += (i ? "," : "") + std::to_string(s[i]); }
			return r + "]";
		}

		std::string dir_;
		float atol_, rtol_;
		std::map<std::string, Entry> entries_;
	};
}