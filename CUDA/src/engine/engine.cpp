#include "engine.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>
#include <iomanip>

//Ensure #define ONNX_ML is defined in configuration to include the ONNX ML operators in the generated protobuf code
#include <onnx/onnx_pb.h>
#include <cuda_runtime.h>

#include "utility.h"
#include "tensor.h"
#include "../kernels/kernel_utility.h"
#include "op.h"
#include "golden.h"

namespace ohy {

	/// <summary>
	/// Inference instance
	/// </summary>
	class Instance {
	public:
		~Instance();
		std::map<std::string, TensorPtr> env;
		std::vector<std::string> input_names;
		std::vector<std::string> output_names;

		std::unique_ptr<onnx::ModelProto> model;
	};

	Engine::Engine() = default;
	Engine::~Engine() {
		Shutdown();
	}

	void Engine::Initialize(const std::string& modelPath) {

		std::filesystem::path abs_path = std::filesystem::absolute(modelPath);
		if (!std::filesystem::exists(abs_path)) {
			throw std::runtime_error("Model file not found: " + abs_path.string());
		}

		std::cout << "Initializing engine with model: " << abs_path << std::endl;
		
		instance.reset(new Instance());
		auto tmp = std::make_unique<onnx::ModelProto>();
		instance->model.swap(tmp);
		std::ifstream f(abs_path, std::ios::binary);
		if (!instance->model->ParseFromIstream(&f)) {
			throw std::runtime_error("bad onnx file");
		}

		// Extracting the graph structure
		const onnx::GraphProto& graph = instance->model->graph();
		std::cout << "Graph Name: " << graph.name() << std::endl;
		std::cout << "Number of Nodes (Layers): " << graph.node_size() << std::endl;

		std::cout << "Inputs:" << std::endl;
		for (int i = 0; i < graph.input_size(); ++i) {
			instance->input_names.push_back(graph.input(i).name());
			std::cout << "  - " << graph.input(i).name() << std::endl;
		}

		// List the output layer names
		std::cout << "Outputs:" << std::endl;
		for (int i = 0; i < graph.output_size(); ++i) {
			instance->output_names.push_back(graph.output(i).name());
			std::cout << "  - " << graph.output(i).name() << std::endl;
		}

		std::cout << "\nDetails:" << std::endl;

		std::set<std::string> host_initializers;
		// Create consumer map from nodes to get inputs that need to be stored in host side
		for (const auto& node : graph.node()) {
			// 479 nodes, already in execution order
			std::cout << utility::ToString(node) << std::endl;

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

		std::cout << "Host initializer: " << host_initializers.size() << std::endl;

		// Initialize
		for (auto& init : graph.initializer()) {
			//std::cout << utility::ToString(init) << std::endl;
			std::cout << utility::ToString(init) << std::endl;

			Location bufferLoc = Location::Device;
			if (host_initializers.contains(init.name())) {
				bufferLoc = Location::Host;
			}
			instance->env[init.name()] = CreateTensorFromONNX(&init, bufferLoc);
		}


		InitDispatch();

		isInitialized = true;
	}


	void Engine::Shutdown() {
		if (!isInitialized) {
			return;
		}
		dispatch.clear();
		instance.reset();
		isInitialized = false;
	}

	std::string Engine::ProcessImage(const std::string& imagePath) {

		// Check if the engine is initialized
		if (!isInitialized) {
			throw std::runtime_error("Engine not initialized. Call Initialize() first.");
		}

		// Check if image file exists
		std::filesystem::path image_abs_path = std::filesystem::absolute(imagePath);
		if (!std::filesystem::exists(image_abs_path)) {
			throw std::runtime_error("Image file not found: " + image_abs_path.string());
		}

		std::cout << "Processing image: " << image_abs_path << std::endl;

		// Check if the model has input names
		if (instance->input_names.empty()) {
			throw std::runtime_error("No input names found in the model.");
		}
		

		// TODO: Load the image, preprocess it, and convert it to the appropriate format for the model.
		//instance.env[instance.input_names[0]] = CreateUnit();

		// Temporarily uses golden input
		std::vector<float> host(4 * 3 * 224 * 224);
		//"python/export/out/golden/pixel_values.bin"
		std::ifstream f(imagePath, std::ios::binary);
		f.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float));

		// Copy the input data to the GPU
		if (golden) {
			std::vector<char> bytes = golden->Load(instance->input_names[0]);   // same bytes ORT ran on
			const Golden::Entry* e = golden->Find(instance->input_names[0]);
			auto input_unit = CreateTensor(Location::Device, DataType::FLOAT32, e->shape, bytes.data(), bytes.size());
			instance->env[instance->input_names[0]] = std::move(input_unit);
		}
		else {
			auto input_unit = std::make_shared<Tensor>();
			input_unit->dtype = DataType::FLOAT32;
			input_unit->storage = std::make_shared<Buffer>(Location::Device, reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
			input_unit->shape = { 4, 3, 224, 224 }; // Assuming the input shape is [batch_size, channels, height, width]
			instance->env[instance->input_names[0]] = std::move(input_unit);
		}

		bool debug = false;
		bool onNodeDone = true; // set to true for per-node verification mode
#ifdef _DEBUG
		debug = true;
#endif // DEBUG

		auto& graph = instance->model->graph();

		OpContext ctx;
		for (int i = 0; i < graph.node_size(); ++i) {
			const auto& node = graph.node(i);
			std::vector<TensorPtr> in;
			for (const auto& name : node.input()) in.push_back(instance->env.at(name));

			if (dispatch.contains(node.op_type()) == false) {
				throw std::runtime_error("Unsupported operation: " + node.op_type());
			}
			auto out = dispatch.at(node.op_type())(&node, in, ctx);
			if (golden) {
				for (int i = 0; i < node.output_size(); ++i) {
					if (!golden->Check(node.output(i), node.op_type(), *out[i], ctx.stream)) {
						//throw std::runtime_error("Golden mismatch at node " + node.name());   // stop at first failure
					}
				}
			}

			for (int k = 0; k < node.output_size(); ++k) instance->env[node.output(k)] = out[k];

			if (debug) {                                  // per-node verification mode
				CUDA_CHECK(cudaDeviceSynchronize());      // async errors surface HERE, at the right node
				//if (onNodeDone) onNodeDone(i, node, out); // test code compares with tensors/NNNN_*.bin
			}
		}


		return "Identified Car Model: ExampleModel";
	}

	bool Engine::InitDispatch() {
		dispatch.clear();

		// Register dispatch functions for each supported operation
		dispatch["Conv"] = OpConv;
		dispatch["Reshape"] = OpReshape;
		dispatch["Transpose"] = OpTranspose;
		dispatch["Concat"] = OpConcat;
		dispatch["Add"] = OpAdd;
		dispatch["LayerNormalization"] = OpLayerNormalization;
		dispatch["MatMul"] = OpMatMul;
		dispatch["Unsqueeze"] = OpUnsqueeze;
		dispatch["Squeeze"] = OpSqueeze;
		dispatch["Sigmoid"] = OpSigmoid;
		dispatch["Gather"] = OpGather;
		dispatch["Mul"] = OpMul;
		// ... all 14. Anything else -> dispatch.at() throws -> "fail loudly"

		std::cout << "Initialized dispatch for supported operations." << std::endl;

		// Check how many ops are not implemented
		std::map<std::string, int> nodeCounts;
		const onnx::GraphProto& graph = instance->model->graph();
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
		std::cout << std::setprecision(2) << "Implemented operations: " << okOp << "/" << okOp + notOkOp
			<< " (" << (float)okOp / (float)(okOp + notOkOp) * 100.0f << "%)" << std::endl
			<< "Implemented nodes: " << okNodes << "/" << okNodes + notOkNode
			<< " (" << (float)okNodes / (float)(okNodes + notOkNode) * 100.0f << "%)" << std::endl;

		return true;
	}

	TensorPtr Engine::CreateTensorFromONNX(const onnx::TensorProto* tensorProto,
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

	void Engine::InitializeOutputCheck(const std::string& dir) {
		golden = std::make_unique<Golden>(dir);
		std::cout << "Loaded golden at " << dir << std::endl;
	}

	Instance::~Instance() = default;
}