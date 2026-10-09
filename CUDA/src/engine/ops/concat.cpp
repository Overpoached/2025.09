#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	namespace {
		size_t ElemSize(DataType t) {
			switch (t) {
			case DataType::FLOAT32: return sizeof(float);
			case DataType::INT64:   return sizeof(int64_t);
			}
			throw std::runtime_error("Concat: unsupported dtype");
		}
	}

	std::vector<TensorPtr> OpConcat(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// Concat takes 1..N inputs. All must share rank and dtype, and match on every dim except `axis`.
		if (in.empty()) {
			throw std::runtime_error("Concat: expected at least 1 input, node " + node->name());
		}

		const Tensor& ref = *in[0];
		const int64_t rank = static_cast<int64_t>(ref.shape.size());

		constexpr int64_t kMissing = std::numeric_limits<int64_t>::min();
		int64_t axis = utility::AttrInt(*node, "axis", kMissing);
		if (axis == kMissing) {
			throw std::runtime_error("Concat: missing required attribute axis, node " + node->name());
		}
		if (axis < 0) {
			axis += rank;   // ONNX allows negative axis, counted from the back
		}
		if (axis < 0 || axis >= rank) {
			throw std::runtime_error("Concat: axis out of range, node " + node->name());
		}

		const Location loc = ref.storage->location;
		const DataType dtype = ref.dtype;
		std::vector<int64_t> outShape = ref.shape;
		outShape[axis] = 0;

		for (const auto& t : in) {
			if (t->storage->location != loc) {
				throw std::runtime_error("Concat: inputs on mixed host/device, node " + node->name());
			}
			if (t->dtype != dtype) {
				throw std::runtime_error("Concat: inputs have mixed dtypes, node " + node->name());
			}
			if (static_cast<int64_t>(t->shape.size()) != rank) {
				throw std::runtime_error("Concat: inputs have different rank, node " + node->name());
			}
			for (int64_t d = 0; d < rank; ++d) {
				if (d != axis && t->shape[d] != ref.shape[d]) {
					throw std::runtime_error("Concat: non-axis dimension mismatch " + utility::ToString(t->shape)
						+ " vs " + utility::ToString(ref.shape) + ", node " + node->name());
				}
			}
			outShape[axis] += t->shape[axis];
		}

		// View every tensor as [outer, shape[axis] * inner]. Each input fills a column band of the output.
		int64_t outer = 1;
		int64_t inner = 1;
		for (int64_t d = 0; d < axis; ++d) { outer *= outShape[d]; }
		for (int64_t d = axis + 1; d < rank; ++d) { inner *= outShape[d]; }

		const size_t elem = ElemSize(dtype);
		const size_t dstPitch = static_cast<size_t>(outShape[axis] * inner) * elem;   // bytes per output row
		auto out = CreateTensor(loc, dtype, outShape, nullptr, static_cast<size_t>(outer) * dstPitch);

		size_t dstOffset = 0;   // byte offset of this input's band within an output row
		for (const auto& t : in) {
			const size_t width = static_cast<size_t>(t->shape[axis] * inner) * elem;   // bytes per input row
			if (width == 0 || outer == 0) {
				continue;   // zero-sized input contributes nothing
			}

			if (loc == Location::Device) {
				cudaError_t err = cudaMemcpy2DAsync(
					out->data() + dstOffset, dstPitch,   // dst, dst pitch
					t->data(), width,                    // src, src pitch (input rows are packed)
					width, static_cast<size_t>(outer),   // bytes per row, number of rows
					cudaMemcpyDeviceToDevice, ctx.stream);
				if (err != cudaSuccess) {
					throw std::runtime_error(std::string("Concat: cudaMemcpy2DAsync failed: ")
						+ cudaGetErrorString(err) + ", node " + node->name());
				}
			}
			else {
				for (int64_t o = 0; o < outer; ++o) {
					std::memcpy(out->data() + o * dstPitch + dstOffset, t->data() + o * width, width);
				}
			}
			dstOffset += width;
		}

		return { out };
	}
}