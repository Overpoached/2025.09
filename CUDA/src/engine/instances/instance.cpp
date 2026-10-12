#include "instance.h"

#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <onnx/onnx_pb.h>

#include "log.h"
#include "compare.h"
#include "golden.h"
#include "crud.h"
#include "infer.h"
#include "../utility.h"
#include "../preprocess.h"
#include "../../kernels/kernel_utility.h"

namespace ohy {

	void Instance::LoadModel(const std::string& modelPath)
	{
		std::filesystem::path abs_path = std::filesystem::absolute(modelPath);
		if (!std::filesystem::exists(abs_path)) {
			throw std::runtime_error("Model file not found: " + abs_path.string());
		}

		logging::Infof("Initializing engine with model: %s", abs_path.string().c_str());

		auto tmp = std::make_unique<onnx::ModelProto>();
		model.swap(tmp);

		std::ifstream f(abs_path, std::ios::binary);
		if (!model->ParseFromIstream(&f)) {
			throw std::runtime_error("bad onnx file");
		}

		// Extracting the graph structure
		const onnx::GraphProto& graph = model->graph();
		logging::Debugf("Graph Name: %s", graph.name().c_str());
		logging::Debugf("Number of Nodes (Layers): %d", graph.node_size());

		logging::Debugf("Inputs: ");
		for (int i = 0; i < graph.input_size(); ++i) {
			input_names.push_back(graph.input(i).name());
			logging::Debugf("  - %s", graph.input(i).name().c_str());
		}

		// List the output layer names
		logging::Debugf("Outputs: ");
		for (int i = 0; i < graph.output_size(); ++i) {
			output_names.push_back(graph.output(i).name());
			logging::Debugf("  - %s", graph.output(i).name().c_str());
		}

		logging::Debugf("\nDetails: ");

		std::set<std::string> host_initializers;
		// Create consumer map from nodes to get inputs that need to be stored in host side
		for (const auto& node : graph.node()) {
			// 479 nodes, already in execution order
			logging::Debugf("%s", utility::ToString(node).c_str());

			if (ohy::host_op_index_map.contains(node.op_type())) {
				auto indices = ohy::host_op_index_map.at(node.op_type());
				for (size_t i = 0; i < indices.size(); ++i) {
					if (i >= node.input_size()) {
						throw std::runtime_error("Could not get input index from node. " +
							node.op_type() + ": " +
							node.name() + "-" +
							node.input(static_cast<int>(i)) + "-" +
							std::to_string(i));
					}
					host_initializers.insert(node.input(static_cast<int>(indices[i])));
				}
			}
			else {
				// ensure inputs all not in host init set
				for (int i = 0; i < node.input_size(); ++i) {
					if (host_initializers.contains(node.input(i))) {
						throw std::runtime_error("Device data in host list. " +
							node.op_type() + ": " +
							node.name() + "-" +
							node.input(static_cast<int>(i)) + "-" +
							std::to_string(i));
					}
				}
			}
		}

		logging::Debugf("Host initializer: %zu", host_initializers.size());

		// Initialize
		for (auto& init : graph.initializer()) {
			logging::Debug("{}", utility::ToString(init));

			Location bufferLoc = Location::Device;
			if (host_initializers.contains(init.name())) {
				bufferLoc = Location::Host;
			}
			env[init.name()] = CreateTensorFromONNX(&init, bufferLoc);
		}

	}
	void Instance::Initialize(const InitArg& initArg)
	{
		LoadModel(initArg.modelPath);
		InitSelf(initArg);
	}
	void Instance::Run(const RunArg& runArg)
	{
		RunSelf(runArg);
	}



	TensorPtr Instance::CreateTensorFromONNX(const onnx::TensorProto* tensorProto,
		Location bufferLoc) const
	{
		if (tensorProto == nullptr) {
			throw std::runtime_error("TensorProto is null");
		}
		auto shape = std::vector<int64_t>(tensorProto->dims().begin(), tensorProto->dims().end());
		auto host_i64 = std::vector<int64_t>(tensorProto->int64_data().begin(), tensorProto->int64_data().end());
		auto type = utility::ONNXDataTypeToDataType(tensorProto->data_type());
		auto ret = ohy::CreateTensor(bufferLoc, type, shape, tensorProto->raw_data().data(), tensorProto->raw_data().size());
		return ret;
	}


	std::map<std::string, TensorPtr> Instance::RunGraph(const RunArg& runArg, OpContext& ctx)
	{
		auto& dispatch = runArg.dispatch;

		auto& graph = model->graph();
		for (int i = 0; i < graph.node_size(); ++i) {
			const auto& node = graph.node(i);
			std::vector<TensorPtr> in;
			for (const auto& name : node.input()) in.push_back(env.at(name));

			if (dispatch.contains(node.op_type()) == false) {
				throw std::runtime_error("Unsupported operation: " + node.op_type());
			}
			auto out = dispatch.at(node.op_type())(&node, in, ctx);

			for (int k = 0; k < node.output_size(); ++k) env[node.output(k)] = out[k];

														// per-node verification mode
			CUDA_CHECK(cudaDeviceSynchronize());      // async errors surface HERE, at the right node
			
		}

		std::map<std::string, TensorPtr> ret;
		for (auto& outName : output_names) {
			if (!env.contains(outName)) {
				throw std::runtime_error("Missing output: " + outName);
			}
			ret[outName] = env.at(outName);
		}
		
		return ret;
	}

	std::vector<std::vector<float>> Instance::EmbedImages(const std::vector<std::string>& imagePaths,
		const RunArg& runArg)
	{
		if (input_names.empty() || output_names.empty()) {
			throw std::runtime_error("EmbedImages: model has no inputs or outputs");
		}

		const std::vector<int64_t> inShape = InputShape();
		const int64_t batch = inShape[0];
		const int64_t slot = inShape[1] * inShape[2] * inShape[3];

		PreprocessConfig cfg;
		cfg.size = static_cast<int>(inShape[3]);
		OpContext ctx;

		auto input = std::make_shared<Tensor>();
		input->dtype = DataType::FLOAT32;
		input->shape = inShape;
		input->storage = std::make_shared<Buffer>(Location::Device, input->numel() * sizeof(float));

		std::vector<std::vector<float>> embeds;
		embeds.reserve(imagePaths.size());
		for (size_t start = 0; start < imagePaths.size(); start += static_cast<size_t>(batch)) {
			const int64_t used = std::min<int64_t>(batch, static_cast<int64_t>(imagePaths.size() - start));
			for (int64_t n = 0; n < used; ++n) {
				logging::Infof("Embedding %s", imagePaths[start + n]);
				PreprocessImage(ReadFileBytes(imagePaths[start + n]), cfg, input->f32() + n * slot, ctx.stream);
			}
			// Fixed batch: pad unused slots with copies of slot 0 and ignore their outputs.
			for (int64_t n = used; n < batch; ++n) {
				CUDA_CHECK(cudaMemcpyAsync(input->f32() + n * slot, input->f32(), slot * sizeof(float),
					cudaMemcpyDeviceToDevice, ctx.stream));
			}
			env[input_names[0]] = input;

			auto outMap = RunGraph(runArg, ctx);
			TensorPtr out = outMap.at(output_names[0]);                // [batch, dim]
			const int64_t dim = out->numel() / batch;
			std::vector<float> host(out->numel());
			CUDA_CHECK(cudaMemcpyAsync(host.data(), out->f32(), host.size() * sizeof(float),
				cudaMemcpyDeviceToHost, ctx.stream));
			CUDA_CHECK(cudaStreamSynchronize(ctx.stream));       // also makes `input` safe to overwrite next chunk

			for (int64_t n = 0; n < used; ++n) {
				embeds.emplace_back(host.begin() + n * dim, host.begin() + (n + 1) * dim);
			}
		}
		return embeds;
	}

	std::vector<int64_t> Instance::InputShape(int index ) const {
		// The dynamo export bakes the batch size in, so every dim is fixed.
		std::vector<int64_t> shape;
		for (const auto& d : model->graph().input(index).type().tensor_type().shape().dim()) {
			shape.push_back(d.dim_value());
		}
		return shape;
	}

	std::unique_ptr<Instance> MakeInstance(InstanceType type) {
		switch (type) {
		case InstanceType::GOLDEN:
			return std::make_unique<Golden>();
		case InstanceType::COMPARE:
			return std::make_unique<CompareInstance>();
		case InstanceType::CRUD:
			return std::make_unique<CRUDInstance>();
		default:
			throw std::runtime_error("MakeInstance: Unsupported instance type");
		}
		return nullptr;
	}
}