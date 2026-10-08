#include "utility.h"

#include <stdexcept>
#include <string>
#include <sstream>

namespace ohy::utility {

	DataType ONNXDataTypeToDataType(onnx::TensorProto_DataType onnxType) {
		switch (onnxType) {
		case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
			return DataType::FLOAT32;
		case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
			return DataType::INT64;
		default:
			throw std::runtime_error("Unsupported ONNX data type: " + std::to_string(onnxType));
		}
	}

	DataType ONNXDataTypeToDataType(int32_t onnxType) {
		return ONNXDataTypeToDataType(static_cast<onnx::TensorProto_DataType>(onnxType));
	}

	int64_t              AttrInt(const onnx::NodeProto& node, const char* name, int64_t def) {
		const auto& attrs = node.attribute();
		for (const auto& attr : attrs) {
			if (attr.name() == name) {
				if (attr.type() == onnx::AttributeProto_AttributeType_INT) {
					return attr.i();
				}
				else {
					throw std::runtime_error("Attribute " + std::string(name) + " is not of type INT");
				}
			}
		}
		return def; // return default if not found
	}
	float                AttrFloat(const onnx::NodeProto& node, const char* name, float def) {

		const auto& attrs = node.attribute();
		for (const auto& attr : attrs) {
			if (attr.name() == name) {
				if (attr.type() == onnx::AttributeProto_AttributeType_FLOAT) {
					return attr.f();
				}
				else {
					throw std::runtime_error("Attribute " + std::string(name) + " is not of type FLOAT");
				}
			}
		}
		return def; // return default if not found
	}
	std::vector<int64_t> AttrInts(const onnx::NodeProto& node, const char* name) {
		const auto& attrs = node.attribute();
		for (const auto& attr : attrs) {
			if (attr.name() == name) {
				if (attr.type() == onnx::AttributeProto_AttributeType_INTS) {
					std::vector<int64_t> result;
					for (int i = 0; i < attr.ints_size(); ++i) {
						result.push_back(attr.ints(i));
					}
					return result;
				}
				else {
					throw std::runtime_error("Attribute " + std::string(name) + " is not of type INTS");
				}
			}
		}
		throw std::runtime_error("Attribute " + std::string(name) + " not found");
	}

	std::string ToString(const onnx::AttributeProto& attr, uint32_t pad) {
		std::stringstream ss;
		std::string padstr(static_cast<size_t>(pad), ' ');
		ss << padstr << "Attribute: " << attr.name();
		ss << "\n" << padstr << "  Type: " << static_cast<int32_t>(attr.type());
		ss << "\n" << padstr << "    [";
		switch (attr.type()) {
		case onnx::AttributeProto_AttributeType_INT: {
			ss << attr.i();
		} break;
		case onnx::AttributeProto_AttributeType_INTS: {
			for (int i = 0; i < attr.ints_size(); ++i) {
				if (i > 0) {
					ss << ", ";
				}
				ss << attr.ints(i);
			}
		} break;
		case onnx::AttributeProto_AttributeType_FLOAT: {
			ss << attr.f();
		} break;
		case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING: {
			ss << attr.s();
		} break;
		default: {
			ss << "Unsupported type";
		}
		}
		ss << "]";
		return ss.str();
	}
	std::string ToString(const onnx::NodeProto& node, uint32_t pad) {
		std::stringstream ss;
		std::string padstr(static_cast<size_t>(pad), ' ');
		ss << padstr << "Node: " << node.name()
			<< "\n" << padstr << "  Type: " << node.op_type()
			<< "\n" << padstr << "  Inputs: " << node.input_size()
			<< "\n" << padstr << "  Outputs: " << node.output_size()
			<< "\n" << padstr << "  Attributes: " << node.attribute_size()
			<< "\n" << padstr << "  [";
		for (int i = 0; i < node.attribute_size(); ++i) {
			ss << "\n" << padstr << ToString(node.attribute(i), pad + 2 + 2);
		}
		ss << "\n" << padstr << "  ]";
		return ss.str();
	}
	std::string ToString(const onnx::TensorProto& tensor, uint32_t pad) {
		std::stringstream ss;
		std::string padstr(static_cast<size_t>(pad), ' ');
		ss << padstr << "Tensor: " << tensor.name();
		return ss.str();
	}
}