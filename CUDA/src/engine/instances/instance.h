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

	enum class CrudOp {
		Add,      // store new images, skip files already stored
		Update,   // replace everything stored under the labels of the given images
		Delete,   // delete every embedding under the given labels, or everything
		Read      // count embeddings per label
	};

	struct InitArg {
		std::string modelPath;
		std::string goldenDirPath;
		std::string storageDir;
		InstanceType type;
		bool loadModel = true;                // false for work that never runs the graph (CRUD read/delete)
	};
	struct RunArg {
		std::vector<std::string> imagePaths;  // CRUD: files or directories (searched recursively)
		std::vector<std::string> labels;      // CRUD add/update: none (parent dir name), one for all, or one per imagePaths entry
		                                      // CRUD delete/read: the labels to act on
		std::string runBy;
		CrudOp crudOp = CrudOp::Add;
		bool all = false;                     // CRUD delete: every label
		int topK = 5;                         // INFER: neighbours that vote on each image's label

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