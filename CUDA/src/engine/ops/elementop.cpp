#include "../op.h"

// elementwise operations: Add, Sub, Mul, Div, Pow, Min, Max, And, Or, Xor
#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>

namespace ohy {
    std::vector<TensorPtr> OpSigmoid(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {
        if (in.size() != 1 || in[0]->dtype != DataType::FLOAT32)
            throw std::runtime_error("Sigmoid: expected 1 float32 input, node " + node->name());
        auto out = CreateTensor(DataType::FLOAT32, in[0]->shape);   // same shape as input; allocates
        kernels::Sigmoid(in[0]->f32(), out->f32(), out->numel(), ctx.stream);
        return { out };
    }

    std::vector<TensorPtr> OpAdd(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {
        if (in.size() != 2 || in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32)
            throw std::runtime_error("Add: expected 2 float32 inputs, node " + node->name());
        auto out = CreateTensor(DataType::FLOAT32, in[0]->shape);   // same shape as input; allocates
        kernels::Add(in[0]->f32(), in[1]->f32(), out->f32(), out->numel(), ctx.stream);
        return { out };
    }
}