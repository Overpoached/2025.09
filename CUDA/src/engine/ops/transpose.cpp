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
			throw std::runtime_error("Transpose: expected 1 inputs (data), node " + node->name());
		}

		auto perms = utility::AttrInts(*node, "perm");
		if (perms.size() == 0) {
			throw std::runtime_error("Transpose: perm attribute size is 0, node " + node->name());
		}
		if (in[0]->shape.size() != perms.size()) {
			throw std::runtime_error("Transpose: input axis count != perm size, node " + node->name());
		}

		std::vector<int64_t> newShape(perms.size());
		for (size_t i = 0; i < in[0]->shape.size(); ++i) {
			newShape[i] = in[0]->shape[perms[i]];
		}

		auto out = CreateTensor(Location::Device, DataType::FLOAT32, newShape, in[0]->data(), in[0]->numel() * sizeof(float)); // shape of output influenced by stride
		return { out };
	}
}