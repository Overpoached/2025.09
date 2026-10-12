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
#include "instances/golden.h"
#include "preprocess.h"
#include "log.h"

namespace ohy {


	Engine::Engine() = default;
	Engine::~Engine() {
		Shutdown();
	}

	void Engine::Initialize(const InitArg& initArg) {

		instance = std::move(MakeInstance(initArg.type));
		instance->Initialize(initArg);

		InitDispatch();

		isInitialized = true;
	}


	void Engine::Run(const RunArg& runArg) {

		if (!instance) throw std::runtime_error("Empty instance");
		if (!isInitialized) {
			throw std::runtime_error("Engine not initialized. Call Initialize() first.");
		}

		RunArg arg = runArg;
		arg.dispatch = dispatch;
		instance->Run(arg);
	}

	void Engine::Shutdown() {
		if (!isInitialized) {
			return;
		}
		dispatch.clear();
		instance.reset();
		isInitialized = false;
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
		dispatch["Softmax"] = OpSoftmax;
		dispatch["Gemm"] = OpGEMM;
		// ... all 14. Anything else -> dispatch.at() throws -> "fail loudly"

		logging::Infof("Initialized dispatch for %zu supported operations.", dispatch.size());

		return true;
	}

}