#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpTranspose(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// 1 input, 1 is data tensor
		if (in.size() != 1) {
			throw std::runtime_error("Transpose: expected 1 input (data), node " + node->name());
		}
		if (in[0]->dtype != DataType::FLOAT32) {
			throw std::runtime_error("Transpose: expected float32 input, node " + node->name());
		}
		if (in[0]->storage->location != Location::Device) {
			throw std::runtime_error("Transpose: expected device input, node " + node->name());
		}

		const auto& inShape = in[0]->shape;
		const int64_t rank = static_cast<int64_t>(inShape.size());
		auto perms = utility::AttrInts(*node, "perm");

		if (static_cast<int64_t>(perms.size()) != rank) {
			throw std::runtime_error("Transpose: input axis count != perm size, node " + node->name());
		}
		if (rank > kernels::kMaxTransposeRank) {
			throw std::runtime_error("Transpose: rank " + std::to_string(rank) + " exceeds max, node " + node->name());
		}

		// Row-major strides of the input
		std::vector<int64_t> inStrides(rank, 1);
		for (int64_t d = rank - 2; d >= 0; --d) {
			inStrides[d] = inStrides[d + 1] * inShape[d + 1];
		}

		kernels::TransposeShape shape{};
		shape.rank = static_cast<int32_t>(rank);
		shape.n = in[0]->numel();
		std::vector<int64_t> newShape(rank);
		for (int64_t i = 0; i < rank; ++i) {
			newShape[i] = inShape[perms[i]];
			shape.outDims[i] = newShape[i];
			shape.inStrides[i] = inStrides[perms[i]];   // walking output axis i moves along input axis perm[i]
		}

		auto out = CreateTensor(Location::Device, DataType::FLOAT32, newShape, nullptr, shape.n * sizeof(float));
		kernels::Transpose(in[0]->f32(), out->f32(), shape, ctx.stream);
		return { out };
	}
}