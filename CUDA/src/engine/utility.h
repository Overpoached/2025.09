#pragma once

#include "datatypes.h"
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>

namespace ohy {
	namespace utility {
		DataType ONNXDataTypeToDataType(onnx::TensorProto_DataType onnxType);
		DataType ONNXDataTypeToDataType(int32_t onnxType);

		int64_t              AttrInt(const onnx::NodeProto&, const char* name, int64_t def);
		float                AttrFloat(const onnx::NodeProto&, const char* name, float def);
		std::vector<int64_t> AttrInts(const onnx::NodeProto&, const char* name);    // throws if missing

		template <typename T>
		std::string ToString(const std::vector<T>& vec, uint32_t pad = 0) {
			std::string ret = "[";
			for (auto& v : vec) {
				if (ret.size() > 1) { ret += ", "; }
				ret += std::to_string(v);
			}
			return ret + "]";
		}

		std::string ToString(const onnx::AttributeProto& attr, uint32_t pad = 0);
		std::string ToString(const onnx::NodeProto& node, uint32_t pad = 0);
		std::string ToString(const onnx::TensorProto& tensor, uint32_t pad = 0);
	}
}