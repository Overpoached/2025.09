#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpGather(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// 2 inputs: data, indices
		if (in.size() != 2 || !in[0] || !in[1]) {
			throw std::runtime_error("Gather: expected 2 inputs (data, indices), node " + node->name());
		}
		const TensorPtr& data = in[0];
		const TensorPtr& indices = in[1];

		if (data->dtype != DataType::FLOAT32 || data->storage->location != Location::Device) {
			throw std::runtime_error("Gather: expected float32 device data, node " + node->name());
		}
		if (indices->dtype != DataType::INT64) {
			throw std::runtime_error("Gather: expected int64 indices, node " + node->name());
		}
		if (indices->storage->location != Location::Device) {
			throw std::runtime_error("Gather: indices are not on device, node " + node->name());
		}

		const auto& shape = data->shape;
		const int64_t rank = static_cast<int64_t>(shape.size());
		if (rank == 0) {
			throw std::runtime_error("Gather: data must have rank >= 1, node " + node->name());
		}
		int64_t axis = utility::AttrInt(*node, "axis", 0);
		if (axis < 0) {
			axis += rank;
		}
		if (axis < 0 || axis >= rank) {
			throw std::runtime_error("Gather: axis out of range, node " + node->name());
		}

		kernels::GatherShape s{};
		s.outer = 1;
		s.inner = 1;
		for (int64_t d = 0; d < axis; ++d) { s.outer *= shape[d]; }
		for (int64_t d = axis + 1; d < rank; ++d) { s.inner *= shape[d]; }
		s.axisDim = shape[axis];
		s.numIndices = indices->numel();   // 1 for a scalar index
		s.n = s.outer * s.numIndices * s.inner;

		// data.shape[:axis] + indices.shape + data.shape[axis+1:]  (scalar index -> axis disappears)
		std::vector<int64_t> outShape(shape.begin(), shape.begin() + axis);
		outShape.insert(outShape.end(), indices->shape.begin(), indices->shape.end());
		outShape.insert(outShape.end(), shape.begin() + axis + 1, shape.end());

		auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, s.n * sizeof(float));
		kernels::Gather(data->f32(), indices->i64(), out->f32(), s, ctx.stream);
		return { out };
	}
}