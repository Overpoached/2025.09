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
		InstanceType type;
	};
	struct RunArg {
		std::vector<std::string> imagePaths;
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

		std::map<std::string, TensorPtr> env;
		std::vector<std::string> input_names;
		std::vector<std::string> output_names;

		std::unique_ptr<onnx::ModelProto> model;
	};

	std::unique_ptr<Instance> MakeInstance(InstanceType type);
}