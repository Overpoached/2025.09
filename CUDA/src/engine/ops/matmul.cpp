#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"
#include <algorithm>

namespace ohy {

	std::vector<TensorPtr> OpMatMul(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		if (in.size() != 2 || in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32) {
			throw std::runtime_error("MatMul: expected 2 float32 inputs, node " + node->name());
		}
		if (in[0]->storage->location != Location::Device || in[1]->storage->location != Location::Device) {
			throw std::runtime_error("MatMul: expected device inputs, node " + node->name());
		}

		std::vector<int64_t> outShape;
		kernels::GEMMShape shape = kernels::MakeMatMulShape(in[0]->shape, in[1]->shape, outShape);

		int64_t count = 1;
		for (int64_t v : outShape) count *= v;
		auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, count * sizeof(float));
		kernels::GEMM(in[0]->f32(), in[1]->f32(), nullptr, out->f32(), shape, ctx.stream);
		return { out };
	}
}