#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpGEMM(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
        if ((in.size() != 2 && in.size() != 3) || in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32)
            throw std::runtime_error("GEMM: expected 2 or 3 float32 inputs, node " + node->name());
        if (in[0]->shape.size() != 2 || in[1]->shape.size() != 2)
            throw std::runtime_error("GEMM: expected 2d shape only for input 1 and 2, node " + node->name());

        float alpha = utility::AttrFloat(*node, "alpha", 1.0);
        float beta = utility::AttrFloat(*node, "beta", 1.0);
        int64_t transA = utility::AttrInt(*node, "transA", 0);
        int64_t transB = utility::AttrInt(*node, "transB", 0);
        
        std::vector<int64_t> outShape;
        kernels::BroadcastShape shape = kernels::MakeBroadcast(in[0]->shape, in[1]->shape, outShape);
        auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, shape.n * sizeof(float));
        kernels::GEMMShape gemmShape;
        gemmShape.batch = outShape.size() == 3 ? outShape[0] : 1;
        gemmShape.alpha = alpha;
        gemmShape.beta = beta;
        gemmShape.hasC = in.size() == 3 ? 1 : 0;
        gemmShape.transA = transA;
        gemmShape.transB = transB;
        gemmShape.m = transA ? in[0]->shape[1] : in[0]->shape[0];
        gemmShape.k = transA ? in[0]->shape[0] : in[0]->shape[1];
        gemmShape.n = transB ? in[1]->shape[0] : in[1]->shape[1];
        kernels::GEMM(in[0]->f32(), in[1]->f32(), in.size() == 3 ? in[2]->f32() : nullptr, out->f32(), gemmShape, ctx.stream);
        return { out };
	}
}