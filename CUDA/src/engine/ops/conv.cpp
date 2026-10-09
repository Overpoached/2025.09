#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
    std::vector<TensorPtr> OpConv(const onnx::NodeProto* node,
        const std::vector<TensorPtr>& in, OpContext& ctx) {

        //std::cout << utility::ToString(*node) << std::endl;

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
        const int64_t yw = (xcols + pad_attr[1] + pad_attr[3] - wcols) / stride_attr[1] + 1;
        const int64_t yh = (xrows + pad_attr[0] + pad_attr[2] - wrows) / stride_attr[0] + 1;


        kernels::ConvShape shapes;
        shapes.xbatch = in[0]->shape[0];
        shapes.xchannels = in[0]->shape[1];
        shapes.xrows = xrows;
        shapes.xcols = xcols;
        shapes.wcount = in[1]->shape[0];
        shapes.wchannels = in[1]->shape[1];
        shapes.wrows = wrows;
        shapes.wcols = wcols;
        shapes.ybatch = shapes.xbatch;
        shapes.ychannels = shapes.wcount;
        shapes.yrows = yh;
        shapes.ycols = yw;
        shapes.uppad = pad_attr[0];
        shapes.leftpad = pad_attr[1];
        shapes.downpad = pad_attr[2];
        shapes.rightpad = pad_attr[3];
        shapes.rowstride = stride_attr[0];
        shapes.colstride = stride_attr[1];

        auto out = CreateTensor(Location::Device, DataType::FLOAT32, { shapes.xbatch, shapes.wcount, yh, yw }, nullptr, shapes.xbatch * shapes.wcount * yw * yh * sizeof(float)); // shape of output influenced by stride

        kernels::Conv(in[0]->f32(), in[1]->f32(), out->f32(), shapes, ctx.stream);
        return { out };
    }
}