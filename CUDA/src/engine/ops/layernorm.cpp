#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpLayerNormalization(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// Inputs: X, Scale, optional B
		if (in.size() < 2 || in.size() > 3) {
			throw std::runtime_error("LayerNormalization: expected 2 or 3 inputs (X, Scale, B), node " + node->name());
		}
		for (const auto& t : in) {
			if (t && (t->dtype != DataType::FLOAT32 || t->storage->location != Location::Device)) {
				throw std::runtime_error("LayerNormalization: expected float32 device inputs, node " + node->name());
			}
		}
		// Mean / InvStdDev outputs are training-only; this engine does not produce them
		for (int i = 1; i < node->output_size(); ++i) {
			if (!node->output(i).empty()) {
				throw std::runtime_error("LayerNormalization: optional outputs Mean/InvStdDev not supported, node " + node->name());
			}
		}
		if (utility::AttrInt(*node, "stash_type", 1) != 1) {
			throw std::runtime_error("LayerNormalization: only stash_type=1 (fp32) supported, node " + node->name());
		}

		const TensorPtr& x = in[0];
		const TensorPtr& scale = in[1];
		const TensorPtr bias = (in.size() == 3) ? in[2] : nullptr;

		const int64_t rank = static_cast<int64_t>(x->shape.size());
		int64_t axis = utility::AttrInt(*node, "axis", -1);
		if (axis < 0) {
			axis += rank;
		}
		if (axis < 0 || axis >= rank) {
			throw std::runtime_error("LayerNormalization: axis out of range, node " + node->name());
		}
		const float epsilon = utility::AttrFloat(*node, "epsilon", 1e-5f);

		// Flatten to [rows, cols]: normalise over every dim from axis to the end
		int64_t rows = 1, cols = 1;
		for (int64_t d = 0; d < axis; ++d) { rows *= x->shape[d]; }
		for (int64_t d = axis; d < rank; ++d) { cols *= x->shape[d]; }

		// ONNX allows Scale/B to broadcast; CLIP's are always full [768]. Support only that.
		if (scale->numel() != cols || (bias && bias->numel() != cols)) {
			throw std::runtime_error("LayerNormalization: Scale/B must have " + std::to_string(cols)
				+ " elements, node " + node->name());
		}

		auto out = CreateTensor(Location::Device, DataType::FLOAT32, x->shape, nullptr, x->numel() * sizeof(float));
		kernels::LayerNorm(x->f32(), scale->f32(), bias ? bias->f32() : nullptr, out->f32(),
			rows, cols, epsilon, ctx.stream);
		return { out };
	}
}