#include "../op.h"

#include "../datatypes.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"
#include <algorithm>

namespace ohy {
	std::vector<TensorPtr> OpUnsqueeze(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// Inserts size-1 dims. No data movement: the output aliases the input's storage.
		// Opset >= 13: axes is input 1 (int64, host). Opset < 13: axes is an attribute.
		if (in.empty() || in.size() > 2 || !in[0]) {
			throw std::runtime_error("Unsqueeze: expected 1 or 2 inputs (data, axes), node " + node->name());
		}

		std::vector<int64_t> axes;
		if (in.size() == 2 && in[1]) {
			if (in[1]->dtype != DataType::INT64) {
				throw std::runtime_error("Unsqueeze: axes must be int64, node " + node->name());
			}
			if (in[1]->storage->location != Location::Host) {
				throw std::runtime_error("Unsqueeze: axes data is not in host, node " + node->name());
			}
			const int64_t* p = in[1]->i64();
			axes.assign(p, p + in[1]->numel());
		}
		else {
			axes = utility::AttrInts(*node, "axes");
		}
		if (axes.empty()) {
			throw std::runtime_error("Unsqueeze: axes is empty, node " + node->name());
		}

		const auto& inShape = in[0]->shape;
		const int64_t outRank = static_cast<int64_t>(inShape.size() + axes.size());

		// Axes index the OUTPUT shape, so negatives wrap around outRank
		for (auto& a : axes) {
			if (a < 0) {
				a += outRank;
			}
			if (a < 0 || a >= outRank) {
				throw std::runtime_error("Unsqueeze: axis out of range " + utility::ToString(axes)
					+ " for output rank " + std::to_string(outRank) + ", node " + node->name());
			}
		}
		std::sort(axes.begin(), axes.end());
		if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
			throw std::runtime_error("Unsqueeze: duplicate axes " + utility::ToString(axes) + ", node " + node->name());
		}

		// Insert in ascending order: each axis is then already its final position in the output
		std::vector<int64_t> outShape = inShape;
		for (int64_t a : axes) {
			outShape.insert(outShape.begin() + a, 1);
		}

		auto out = std::make_shared<Tensor>();
		out->dtype = in[0]->dtype;
		out->shape = std::move(outShape);
		out->storage = in[0]->storage;   // alias, zero-copy; keeps whatever location the input has
		return { out };
	}
}