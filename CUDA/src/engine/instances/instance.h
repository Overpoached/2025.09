#pragma once

#include <map>
#include <vector>
#include <memory>
#include <string>

#include "../tensor.h"
#include "../op.h"
#include <onnx/onnx_pb.h>
#include <cuda_runtime.h>


namespace ohy {

	enum class InstanceType {
		INFER,
		GOLDEN,
		COMPARE,
		CRUD
	};

	struct InitArg {
		std::string modelPath;
		std::string goldenDirPath;
		std::string storageDir;
		InstanceType type;
	};
	struct RunArg {
		std::vector<std::string> imagePaths;  // CRUD: files or directories (searched recursively)
		std::vector<std::string> labels;      // CRUD: none (parent dir name), one for all, or one per imagePaths entry
		std::string runBy;

		std::map<std::string, OpFn> dispatch; // Dispatch map for node processing
	};
	/// <summary>
	/// Inference instance interface. 
	/// Similar to scenes, managed by engine
	/// </summary>
	class Instance {
	public:
		virtual ~Instance() = default;

		void Initialize(const InitArg& initArg);
		void Run(const RunArg& runArg);
	protected:
		Instance() = default;
		void LoadModel(const std::string& modelPath);
		virtual void InitSelf(const InitArg& initArg) = 0;
		virtual void RunSelf(const RunArg& runArg) = 0;

		// common helpers
		TensorPtr CreateTensorFromONNX(const onnx::TensorProto* tensorProto,
			Location bufferLoc) const;
		std::vector<int64_t> InputShape(int index = 0) const;
		std::map<std::string, TensorPtr> RunGraph(const RunArg& runArg, OpContext& ctx);
		// Preprocess + run the graph over imagePaths in batch-sized chunks.
		// Returns one raw (un-normalised) embedding per image, in input order.
		std::vector<std::vector<float>> EmbedImages(const std::vector<std::string>& imagePaths,
			const RunArg& runArg);

		std::map<std::string, TensorPtr> env;
		std::vector<std::string> input_names;
		std::vector<std::string> output_names;

		std::unique_ptr<onnx::ModelProto> model;
	};

	std::unique_ptr<Instance> MakeInstance(InstanceType type);
}