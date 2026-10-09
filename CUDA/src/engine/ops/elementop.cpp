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


    // Numpy broadcast of a and b. Fills strides with 0 on broadcast axes. Throws on incompatible shapes.
    kernels::BroadcastShape MakeBroadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
        std::vector<int64_t>& outShape, const onnx::NodeProto* node)
    {
        const int64_t rank = static_cast<int64_t>(std::max(a.size(), b.size()));
        if (rank > kernels::kMaxBroadcastRank) {
            throw std::runtime_error("Broadcast: rank too large, node " + node->name());
        }

        // Right-align: pad the shorter shape with leading 1s
        std::vector<int64_t> pa(rank - a.size(), 1), pb(rank - b.size(), 1);
        pa.insert(pa.end(), a.begin(), a.end());
        pb.insert(pb.end(), b.begin(), b.end());

        kernels::BroadcastShape s{};
        s.rank = static_cast<int32_t>(rank);
        outShape.assign(rank, 1);
        int64_t strideA = 1, strideB = 1;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (pa[d] != pb[d] && pa[d] != 1 && pb[d] != 1) {
                throw std::runtime_error("Broadcast: incompatible shapes " + utility::ToString(a)
                    + " and " + utility::ToString(b) + ", node " + node->name());
            }
            outShape[d] = std::max(pa[d], pb[d]);
            s.outDims[d] = outShape[d];
            s.aStrides[d] = (pa[d] == 1) ? 0 : strideA;   // size-1 axis: repeat the same element
            s.bStrides[d] = (pb[d] == 1) ? 0 : strideB;
            strideA *= pa[d];
            strideB *= pb[d];
        }
        s.n = 1;
        for (int64_t v : outShape) s.n *= v;
        return s;
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
        kernels::BroadcastShape shape = MakeBroadcast(in[0]->shape, in[1]->shape, outShape, node);
        auto out = CreateTensor(Location::Device, DataType::FLOAT32, outShape, nullptr, shape.n * sizeof(float));
        kernels::AddBroadcast(in[0]->f32(), in[1]->f32(), out->f32(), shape, ctx.stream);
        return { out };
    }
}