// golden.h — test/debug only
#pragma once
#include "../tensor.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "instance.h"

namespace ohy {

	class Golden : public Instance {
	public:
		struct Entry { std::string file; std::string dtype; std::vector<int64_t> shape; };

		Golden() = default;

		Golden(std::string dir, float atol = 1e-4f, float rtol = 1e-3f);

		std::vector<char> Load(const std::string& name) const;

		const Entry* Find(const std::string& name) const;

		// Returns true on pass. Prints one line per tensor.
		bool Check(const std::string& name, const std::string& opType, const Tensor& t, cudaStream_t stream) const;

	private:
		virtual void InitSelf(const InitArg& initArg) override;
		virtual void RunSelf(const RunArg& runArg) override;
		void LoadManifest(const std::string& dir);

		static std::vector<char> ToHost(const Tensor& t, cudaStream_t stream, size_t bytes);
		static std::string ToStr(const std::vector<int64_t>& s);

		std::string dir_;
		float atol_ = 1e-4f, rtol_ = 1e-3f;
		std::map<std::string, Entry> entries_;
	};
}