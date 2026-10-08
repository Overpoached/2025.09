#include "../op.h"

// elementwise operations: Add, Sub, Mul, Div, Pow, Min, Max, And, Or, Xor
#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
    std::vector<TensorPtr> OpConv(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {

        std::cout << utility::ToString(*node) << std::endl;

        if (in.size() != 2) {
            throw std::runtime_error("Conv: expected 2 inputs (data and weights), node " + node->name());
        }
		if (in[0]->dtype != DataType::FLOAT32 || in[1]->dtype != DataType::FLOAT32) {
			throw std::runtime_error("Conv: expected float32 inputs, node " + node->name());
		}

        if (in[0]->shape.size() != 4) {
            throw std::runtime_error("Conv: expected in[0] shape/dim size = 4, size " + std::to_string(in[0]->shape.size()));
        }
        
        int64_t xcols = in[0]->shape[3];
        int64_t xrows = in[0]->shape[2];

        if (in[1]->shape.size() != 4) {
            throw std::runtime_error("Conv: expected in[1] shape/dim size = 4, size " + std::to_string(in[1]->shape.size()));
        }


        int64_t wcols = in[1]->shape[3];
        int64_t wrows = in[1]->shape[2];

        auto stride = utility::AttrInts(*node, "strides");
        auto pad = utility::AttrInts(*node, "pads");
        // output dimension
        const int64_t woffset = (wcols / 2) * 2 - pad[1] - pad[3];
        const int64_t hoffset = (wrows / 2) * 2 - pad[0] - pad[2];
        const int64_t yw = (xcols - woffset + stride[1] - 1) / stride[1];
        const int64_t yh = (xrows - hoffset + stride[0] - 1) / stride[0];

        auto out = CreateTensor(DataType::FLOAT32, { yh, yw }); // shape of output influenced by stride

        kernels::Conv(in[0]->f32(), xrows, xcols, in[1]->f32(), wrows, wcols, out->f32(), yh, yw, pad.data(), pad.size(), stride.data(), stride.size(), ctx.stream);
        return { out };
    }
}