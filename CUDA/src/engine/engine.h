#pragma once

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <set>

#include "datatypes.h"
#include "tensor.h"
#include "op.h"

namespace onnx {
	class ModelProto;
	class NodeProto;
	class TensorProto;
}

namespace ohy {

	class Instance;
	class Golden;
	/// <summary>
	/// Represents the main engine for handling image processing and car model identification.
	/// </summary>
	class Engine {
	public:
		Engine();
		~Engine();
		/// <summary>
		/// Initializes the engine with the specified model path.
		/// </summary>
		/// <param name="modelPath">The path to the model file.</param>
		void Initialize(const std::string& modelPath);
		void InitializeOutputCheck(const std::string& dir);
		void Shutdown();
		/// <summary>
		/// Processes the input image and returns the identified car model.
		/// </summary>
		/// <param name="imagePath">The path to the input image.</param>
		/// <returns>The identified car model as a string.</returns>
		std::string ProcessImage(const std::string& imagePath);

	private:

		bool InitDispatch();

		TensorPtr CreateTensorFromONNX(const onnx::TensorProto* tensorProto,
			Location bufferLoc) const;

	private:
		bool isInitialized = false;  // Flag to check if the engine is initialized
		std::unique_ptr< Instance > instance;
		std::map<std::string, OpFn> dispatch; // Dispatch map for node processing
		std::unique_ptr<Golden> golden;
	};


}