#include "../op.h"

// elementwise operations: Add, Sub, Mul, Div, Pow, Min, Max, And, Or, Xor
#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"
#include <algorithm>

namespace ohy {
    std::vector<TensorPtr> OpSigmoid(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {
        if (in.size() != 1 || in[0]->dtype != DataType::FLOAT32)
            throw std::runtime_error("Sigmoid: expected 1 float32 input, node " + node->name());
        auto out = CreateTensor(Location::Device, DataType::FLOAT32, in[0]->shape, nullptr, in[0]->numel() * sizeof(float));   // same shape as input; allocates
        kernels::Sigmoid(in[0]->f32(), out->f32(), out->numel(), ctx.stream);
        return { out };
    }


    std::vector<TensorPtr> OpAdd(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {
        if (in.size() != 2 || in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32)
            throw std::runtime_error("Add: expected 2 float32 inputs, node " + node->name());

        if (in[0]->shape == in[1]->shape) {   // fast path: no broadcasting, no div/mod
            auto out = CreateTensor(Location::Device, DataType::FLOAT32, in[0]->shape, nullptr, in[0]->numel() * sizeof(float));
            kernels::Add(in[0]->f32(), in[1]->f32(), out->f32(), out->numel(), ctx.stream);
            return { out };
        }

        std::vector<int64_t> outShape;
        kernels::BroadcastShape shape = kernels::MakeBroadcast(in[0]->shape, in[1]->shape, outShape);
        auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, shape.n * sizeof(float));
        kernels::Add(in[0]->f32(), in[1]->f32(), out->f32(), shape, ctx.stream);
        return { out };
    }

    std::vector<TensorPtr> OpMul(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {
        if (in.size() != 2 || in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32)
            throw std::runtime_error("Mul: expected 2 float32 inputs, node " + node->name());

        if (in[0]->shape == in[1]->shape) {   // fast path: no broadcasting, no div/mod
            auto out = CreateTensor(Location::Device, DataType::FLOAT32, in[0]->shape, nullptr, in[0]->numel() * sizeof(float));
            kernels::Mul(in[0]->f32(), in[1]->f32(), out->f32(), out->numel(), ctx.stream);
            return { out };
        }

        std::vector<int64_t> outShape;
        kernels::BroadcastShape shape = kernels::MakeBroadcast(in[0]->shape, in[1]->shape, outShape);
        auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, shape.n * sizeof(float));
        kernels::Mul(in[0]->f32(), in[1]->f32(), out->f32(), shape, ctx.stream);
        return { out };
    }
}