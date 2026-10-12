#pragma once

#include "datatypes.h"
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include "common.h"

namespace ohy {
	namespace utility {
		DataType ONNXDataTypeToDataType(onnx::TensorProto_DataType onnxType);
		DataType ONNXDataTypeToDataType(int32_t onnxType);

		int64_t              AttrInt(const onnx::NodeProto&, const char* name, int64_t def);
		float                AttrFloat(const onnx::NodeProto&, const char* name, float def);
		std::vector<int64_t> AttrInts(const onnx::NodeProto&, const char* name);    // throws if missing

		std::string ToString(const onnx::AttributeProto& attr, uint32_t pad = 0);
		std::string ToString(const onnx::NodeProto& node, uint32_t pad = 0);
		std::string ToString(const onnx::TensorProto& tensor, uint32_t pad = 0);
	}
}