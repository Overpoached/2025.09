#include "golden.h"

#include "../../kernels/kernel_utility.h"
#include "../utility.h"
#include "log.h"

namespace ohy {

	Golden::Golden(std::string dir, float atol, float rtol)
		: dir_(std::move(dir)), atol_(atol), rtol_(rtol)
	{
		LoadManifest(dir_);
	}

	std::vector<char> Golden::Load(const std::string& name) const {
		const Entry& e = entries_.at(name);
		std::ifstream f(dir_ + "/" + e.file, std::ios::binary | std::ios::ate);
		if (!f) { throw std::runtime_error("Golden: cannot open " + e.file); }
		std::vector<char> buf(static_cast<size_t>(f.tellg()));
		f.seekg(0);
		f.read(buf.data(), buf.size());
		return buf;
	}

	const Golden::Entry* Golden::Find(const std::string& name) const {
		auto it = entries_.find(name);
		return it == entries_.end() ? nullptr : &it->second;
	}

	bool Golden::Check(const std::string& name, const std::string& opType, const Tensor& t, cudaStream_t stream) const
	{
		const Entry* e = Find(name);
		if (!e) { logging::Infof("[SKIP] %-40s %-10s no golden\n", name.c_str(), opType.c_str()); return true; }

		if (t.shape != e->shape) {
			logging::Infof("[FAIL] %-40s %-10s shape got %s expected %s\n", name.c_str(), opType.c_str(),
				ToStr(t.shape).c_str(), ToStr(e->shape).c_str());
			return false;
		}
		if (t.dtype != DataType::FLOAT32 || e->dtype != "float32") {
			// int64 tensors are shape metadata: compare exactly
			std::vector<char> ref = Load(name);
			std::vector<char> got = ToHost(t, stream, ref.size());
			bool ok = got == ref;
			logging::Infof("[%s] %-40s %-10s int64 exact\n", ok ? "PASS" : "FAIL", name.c_str(), opType.c_str());
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
		std::string logStr = ohy::Format("[%s] %-40s %-10s %-18s maxAbs %.3e maxRel %.3e bad %lld/%zu",
			ok ? "PASS" : "FAIL", name.c_str(), opType.c_str(), ToStr(t.shape).c_str(),
			maxAbs, maxRel, static_cast<long long>(bad), n);
		if (worst >= 0) { logStr += ohy::Format("  worst[%lld] got %.6f ref %.6f", static_cast<long long>(worst), got[worst], ref[worst]); }

		logging::Infof("%s", logStr);
		return ok;
	}

	void Golden::InitSelf(const InitArg& initArg)
	{
		LoadManifest(initArg.goldenDirPath);
	}
	void Golden::RunSelf(const RunArg& runArg)
	{
		auto& dispatch = runArg.dispatch;
		{
			// Check how many ops are not implemented
			std::map<std::string, int> nodeCounts;
			const onnx::GraphProto& graph = model->graph();
			for (const auto& node : graph.node()) {
				if (nodeCounts.contains(node.op_type())) {
					nodeCounts.at(node.op_type())++;
				}
				else {
					nodeCounts[node.op_type()] = 1;
				}
			}

			int okNodes = 0, notOkNode = 0, okOp = 0, notOkOp = 0;
			for (auto& entry : nodeCounts) {
				if (dispatch.contains(entry.first)) {
					okOp++;
					okNodes += entry.second;
				}
				else {
					notOkOp++;
					notOkNode += entry.second;
				}
			}
			logging::Infof("Implemented operations: %d/%d (%.2f%%)\nImplemented nodes:  %d/%d (%.2f%%)\n",
				okOp, okOp + notOkOp, (float)okOp / (float)(okOp + notOkOp) * 100.0f,
				okNodes, okNodes + notOkNode, (float)okNodes / (float)(okNodes + notOkNode) * 100.0f);
		}

		OpContext ctx;

		std::vector<char> bytes = Load(input_names[0]);   // same bytes ORT ran on
		const Golden::Entry* e = Find(input_names[0]);
		env[input_names[0]] =
			CreateTensor(Location::Device, DataType::FLOAT32, e->shape, bytes.data(), bytes.size());

		auto& graph = model->graph();
		for (int i = 0; i < graph.node_size(); ++i) {
			const auto& node = graph.node(i);
			std::vector<TensorPtr> in;
			for (const auto& name : node.input()) in.push_back(env.at(name));

			if (dispatch.contains(node.op_type()) == false) {
				throw std::runtime_error("Unsupported operation: " + node.op_type());
			}
			auto out = dispatch.at(node.op_type())(&node, in, ctx);
			for (int k = 0; k < node.output_size(); ++k) {
				Check(node.output(k), node.op_type(), *out[k], ctx.stream);
			}

			for (int k = 0; k < node.output_size(); ++k) env[node.output(k)] = out[k];

						                                 // per-node verification mode
			CUDA_CHECK(cudaDeviceSynchronize());      // async errors surface HERE, at the right node
		}

		// Output
		std::map<std::string, TensorPtr> outputTensors;
		for (const auto& outputName : output_names) {
			if (!env.contains(outputName)) {
				throw std::runtime_error("Missing output: " + outputName);
			}
			outputTensors[outputName] = env.at(outputName);
		}

		logging::Debug("Output:");
		for (auto& entry : outputTensors) {
			auto tensor = entry.second;
			if (tensor->dtype == ohy::DataType::FLOAT32) {
				std::vector<float> data(tensor->numel());
				cudaMemcpy(data.data(), tensor->storage->ptr, data.size() * sizeof(float), cudaMemcpyDeviceToHost);
				logging::Debugf("%s: %s", entry.first.c_str(), ToString(data).c_str());
			}
			else if (tensor->dtype == ohy::DataType::INT64) {
				std::vector<int64_t> data(tensor->numel());
				cudaMemcpy(data.data(), tensor->storage->ptr, data.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
				logging::Debugf("%s: %s", entry.first.c_str(), ToString(data).c_str());
			}
		}


	}

	void Golden::LoadManifest(const std::string& dir)
	{
		dir_ = dir;
		logging::Infof("Loaded golden at %s", dir.c_str());

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

	std::vector<char> Golden::ToHost(const Tensor& t, cudaStream_t stream, size_t bytes) {
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

	std::string Golden::ToStr(const std::vector<int64_t>& s) {
		std::string r = "[";
		for (size_t i = 0; i < s.size(); ++i) { r += (i ? "," : "") + std::to_string(s[i]); }
		return r + "]";
	}
}