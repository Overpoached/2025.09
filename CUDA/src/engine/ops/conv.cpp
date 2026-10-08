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

        auto stride_attr = utility::AttrInts(*node, "strides");
        auto pad_attr = utility::AttrInts(*node, "pads");
        // output dimension
        const int64_t woffset = (wcols / 2) * 2 - pad_attr[1] - pad_attr[3];
        const int64_t hoffset = (wrows / 2) * 2 - pad_attr[0] - pad_attr[2];
        const int64_t yw = (xcols - woffset + stride_attr[1] - 1) / stride_attr[1];
        const int64_t yh = (xrows - hoffset + stride_attr[0] - 1) / stride_attr[0];

        // TODO: consider copy as value for pad and stride
        auto pad = CreateTensor(DataType::INT64, { 1, 4 }, reinterpret_cast<char*>(pad_attr.data()), pad_attr.size() * sizeof(decltype(pad_attr)::value_type));
        auto stride = CreateTensor(DataType::INT64, { 1, 4 }, reinterpret_cast<char*>(stride_attr.data()), stride_attr.size() * sizeof(decltype(stride_attr)::value_type));

        auto out = CreateTensor(DataType::FLOAT32, { yh, yw }); // shape of output influenced by stride

        kernels::ConvShape shapes;
        shapes.xchannels = 1;
        shapes.xrows = xrows;
        shapes.xcols = xcols;
        shapes.wchannels = 1;
        shapes.wrows = wrows;
        shapes.wcols = wcols;
        shapes.ychannels = 1;
        shapes.yrows = yh;
        shapes.ycols = yw;
        kernels::Conv(in[0]->f32(), in[1]->f32(), out->f32(), shapes, pad->i64(), pad->numel(), stride->i64(), stride->numel(), ctx.stream);
        return { out };
    }
}