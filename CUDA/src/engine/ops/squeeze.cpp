#include "../op.h"

#include "../datatypes.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"
#include <algorithm>

namespace ohy {
	std::vector<TensorPtr> OpSqueeze(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// Remove size-1 dims. No data movement: the output aliases the input's storage.
		if (in.empty() || in.size() > 2 || !in[0]) {
			throw std::runtime_error("Squeeze: expected 1 or 2 inputs (data, axes), node " + node->name());
		}

		std::vector<int64_t> axes;
		if (in.size() == 2 && in[1]) {
			if (in[1]->dtype != DataType::INT64) {
				throw std::runtime_error("Squeeze: axes must be int64, node " + node->name());
			}
			if (in[1]->storage->location != Location::Host) {
				throw std::runtime_error("Squeeze: axes data is not in host, node " + node->name());
			}
			const int64_t* p = in[1]->i64();
			axes.assign(p, p + in[1]->numel());
		}
		else {
			axes = utility::AttrInts(*node, "axes");
		}

		// If axes is not provided, all the single dimensions will be removed from the shape
		const auto& inShape = in[0]->shape;
		if (axes.empty()) {
			std::copy_if(in[0]->shape.begin(), in[0]->shape.end(), axes.end(),
				[](int64_t i) { return i == 1; });
		}

		const int64_t outRank = static_cast<int64_t>(inShape.size());

		// Axes index the OUTPUT shape, so negatives wrap around outRank
		for (auto& a : axes) {
			if (a < 0) {
				a += outRank;
			}
			if (a < 0 || a >= outRank) {
				throw std::runtime_error("Squeeze: axis out of range " + utility::ToString(axes)
					+ " for output rank " + std::to_string(outRank) + ", node " + node->name());
			}
		}
		std::sort(axes.begin(), axes.end(), std::greater<int64_t>());
		if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
			throw std::runtime_error("Squeeze: duplicate axes " + utility::ToString(axes) + ", node " + node->name());
		}

		// Remove in descending order: each axis is then already its final position in the output
		std::vector<int64_t> outShape = inShape;
		for (int64_t a : axes) {
			auto it = outShape.begin() + a;
			if (*it != 1) {
				throw std::runtime_error("Squeeze: axis " + utility::ToString(axes) + " != 1, node " + node->name());
			}
			outShape.erase(outShape.begin() + a);
		}

		auto out = std::make_shared<Tensor>();
		out->dtype = in[0]->dtype;
		out->shape = std::move(outShape);
		out->storage = in[0]->storage;   // alias, zero-copy; keeps whatever location the input has
		return { out };
	}
}