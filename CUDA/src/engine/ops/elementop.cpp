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

    std::vector<TensorPtr> OpSoftmax(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx)
    {
        // Opset >= 13 semantics: softmax along a single axis (default -1).
        // (Opset < 13 flattened everything from `axis` onwards into one row; not supported.)
        if (in.size() != 1 || !in[0]) {
            throw std::runtime_error("Softmax: expected 1 input, node " + node->name());
        }
        const TensorPtr& x = in[0];
        if (x->dtype != DataType::FLOAT32 || x->storage->location != Location::Device) {
            throw std::runtime_error("Softmax: expected float32 device input, node " + node->name());
        }

        const auto& shape = x->shape;
        const int64_t rank = static_cast<int64_t>(shape.size());
        if (rank == 0) {
            throw std::runtime_error("Softmax: input must have rank >= 1, node " + node->name());
        }
        int64_t axis = utility::AttrInt(*node, "axis", -1);
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw std::runtime_error("Softmax: axis out of range, node " + node->name());
        }

        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < axis; ++d) { outer *= shape[d]; }
        for (int64_t d = axis + 1; d < rank; ++d) { inner *= shape[d]; }

        kernels::SoftmaxShape s{};
        s.cols = shape[axis];
        s.inner = inner;
        s.rows = outer * inner;

        auto out = CreateTensor(Location::Device, DataType::FLOAT32, shape, nullptr, x->numel() * sizeof(float));
        kernels::Softmax(x->f32(), out->f32(), s, ctx.stream);
        return { out };
    }
}