#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpGEMM(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
        // ONNX Gemm: Y = alpha * op(A) * op(B) + beta * C
        // A and B are strictly 2-D (no batching); C broadcasts one way into [M, N].
        if (in.size() != 2 && in.size() != 3) {
            throw std::runtime_error("GEMM: expected 2 or 3 inputs, node " + node->name());
        }
        for (const auto& t : in) {
            if (t && (t->dtype != DataType::FLOAT32 || t->storage->location != Location::Device)) {
                throw std::runtime_error("GEMM: expected float32 device inputs, node " + node->name());
            }
        }
        const auto& as = in[0]->shape;
        const auto& bs = in[1]->shape;
        if (as.size() != 2 || bs.size() != 2) {
            throw std::runtime_error("GEMM: expected 2-D A and B, node " + node->name());
        }

        float alpha = utility::AttrFloat(*node, "alpha", 1.0);
        float beta = utility::AttrFloat(*node, "beta", 1.0);
        int64_t transA = utility::AttrInt(*node, "transA", 0);
        int64_t transB = utility::AttrInt(*node, "transB", 0);
        
        const int64_t M = transA ? as[1] : as[0];
        const int64_t K = transA ? as[0] : as[1];
        const int64_t Kb = transB ? bs[1] : bs[0];
        const int64_t N = transB ? bs[0] : bs[1];
        if (K != Kb) {
            throw std::runtime_error("GEMM: inner dims differ " + ToString(as) + (transA ? "^T" : "")
                + " x " + ToString(bs) + (transB ? "^T" : "") + ", node " + node->name());
        }

        kernels::GEMMShape s;
        s.batch = 1;
        s.m = M;
        s.n = N;
        s.k = K;
        s.alpha = alpha;
        s.beta = beta;
        s.transA = transA ? 1 : 0;
        s.transB = transB ? 1 : 0;
        s.batchStrideA = M * K;
        s.batchStrideB = K * N;
        s.batchStrideC = M * N;

        const float* bias = nullptr;
        if (in.size() == 3 && in[2]) {
            // Right-align C's shape against [M, N]; each dim must be 1 or match
            const auto& cs = in[2]->shape;
            if (cs.size() > 2) {
                throw std::runtime_error("GEMM: C must have rank <= 2, node " + node->name());
            }
            const int64_t cm = cs.size() == 2 ? cs[0] : 1;
            const int64_t cn = cs.empty() ? 1 : cs.back();
            if ((cm != 1 && cm != M) || (cn != 1 && cn != N)) {
                throw std::runtime_error("GEMM: C " + ToString(cs) + " does not broadcast to ["
                    + std::to_string(M) + ", " + std::to_string(N) + "], node " + node->name());
            }
            s.biasRowStride = (cm == 1) ? 0 : cn;   // row stride in C's own layout
            s.biasColStride = (cn == 1) ? 0 : 1;
            s.hasC = 1;
            bias = in[2]->f32();
        }

        auto out = CreateTensor(Location::Device, DataType::FLOAT32, { M, N }, nullptr, M * N * sizeof(float));
        kernels::GEMM(in[0]->f32(), in[1]->f32(), bias, out->f32(), s, ctx.stream);
        return { out };
	}
}