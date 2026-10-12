#include "../op.h"

#include "../datatypes.h"
#include "../../kernels/nn/nn.h"
#include <onnx/onnx_pb.h>
#include "../utility.h"

namespace ohy {
	std::vector<TensorPtr> OpReshape(const onnx::NodeProto* node,
		const std::vector<TensorPtr>& in, OpContext& ctx)
	{
		// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. 
		// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor)

		// 2 input, 1 is data tensor, 2 is shape
		if (in.size() != 2) {
			throw std::runtime_error("Reshape: expected 2 inputs (data and shape), node " + node->name());
		}

		if (in[1]->storage->location != Location::Host) {
			throw std::runtime_error("Reshape: shape data is not in host, node " + node->name());
		}

		// Check only at most one dimension is -1
		std::vector<int64_t> newDim(in[1]->numel(), 0);

		const int64_t* shapeData = in[1]->i64();
		const int64_t n = in[1]->numel();
		int64_t inferIndex = -1;
		int64_t dim = 1;
		for (int64_t i = 0; i < n; ++i) {
			if (inferIndex > -1 && shapeData[i] == -1) {
				throw std::runtime_error("Reshape: expected at most one dimension is -1, found 2. node " + node->name());
			}
			else if (shapeData[i] == -1) {
				inferIndex = i;
			}
			else if (shapeData[i] == 0) {
				if (i >= in[0]->shape.size()) {
					throw std::runtime_error("Reshape: could not determine dimension from input data shape. node " + node->name());
				}
				newDim[i] = in[0]->shape[i];
				dim *= newDim[i];
			}
			else {
				newDim[i] = shapeData[i];
				dim *= newDim[i];
			}
		}

		// Calculate infer index
		if (inferIndex > -1)
			newDim[inferIndex] = in[0]->numel() / dim;
		
		auto out = CreateTensor(Location::Device, DataType::FLOAT32, newDim, reinterpret_cast<char*>(in[0]->f32()), in[0]->numel() * sizeof(float)); // shape of output influenced by stride
		return { out };
	}
}