#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include "tensor.h"

namespace onnx {
	class NodeProto;
}

namespace ohy {

	struct OpContext { cudaStream_t stream = nullptr; };

	using OpFn = std::function<std::vector<TensorPtr>(
		const onnx::NodeProto*, const std::vector<TensorPtr>& inputs, OpContext&)>;

	std::vector<TensorPtr> OpSigmoid(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpConv(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpAdd(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpReshape(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpTranspose(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpConcat(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpLayerNormalization(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpMatMul(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpGEMM(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpUnsqueeze(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpSqueeze(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	std::vector<TensorPtr> OpGather(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx);
	// Helpers

}