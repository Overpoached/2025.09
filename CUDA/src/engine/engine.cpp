#include "engine.h"

#include <iostream>
#include <fstream>
#include <filesystem>

#define ONNX_ML // Define ONNX_ML to include the ONNX ML operators in the generated protobuf code
#include <onnx/onnx_pb.h>

namespace ohy {
	Engine::~Engine() {
		Shutdown();
	}

	void Engine::Initialize(const std::string& modelPath) {
		// Here you would load the model from the specified path.
		// For demonstration purposes, we'll just print a message.
		std::filesystem::path abs_path = std::filesystem::absolute(modelPath);
		if (!std::filesystem::exists(abs_path)) {
			throw std::runtime_error("Model file not found: " + abs_path.string());
		}

		std::cout << "Initializing engine with model: " << abs_path << std::endl;

		onnx::ModelProto model;
		std::ifstream f(abs_path, std::ios::binary);
		if (!model.ParseFromIstream(&f)) {
			throw std::runtime_error("bad onnx file");
		}

		// Extracting the graph structure
		const onnx::GraphProto& graph = model.graph();
		std::cout << "Graph Name: " << graph.name() << std::endl;
		std::cout << "Number of Nodes (Layers): " << graph.node_size() << std::endl;

		// List the input layer names
		std::cout << "Inputs:" << std::endl;
		for (int i = 0; i < graph.input_size(); ++i) {
			std::cout << "  - " << graph.input(i).name() << std::endl;
		}		
		
		// List the output layer names
		std::cout << "Outputs:" << std::endl;
		for (int i = 0; i < graph.output_size(); ++i) {
			std::cout << "  - " << graph.output(i).name() << std::endl;
		}

		std::cout << "\nDetails:" << std::endl;
		for (const auto& init : model.graph().initializer()) {
			// weights 
			// init.name(), init.dims(), init.raw_data()  -> compare against manifest "initializers"
			std::cout << "Initializer: " << init.name() << ", dims: " << init.dims().size() << ", raw_data: " << init.raw_data().size() << std::endl;
		}
		for (const auto& node : model.graph().node()) {
			// 479 nodes, already in execution order
			std::cout << "Node: " << node.op_type() << ", inputs: " << node.input().size() << ", outputs: " << node.output().size() << ", attributes: " << node.attribute().size() << std::endl;
		}



		isInitialized = true;
	}


	void Engine::Shutdown() {
		if (!isInitialized) {
			return;
		}

		isInitialized = false;
	}

	std::string Engine::ProcessImage(const std::string& imagePath) {
		if (!isInitialized) {
			throw std::runtime_error("Engine not initialized. Call Initialize() first.");
		}
		// Here you would process the image and identify the car model.
		// For demonstration purposes, we'll just return a dummy string.
		std::cout << "Processing image: " << imagePath << std::endl;
		return "Identified Car Model: ExampleModel";
	}

}