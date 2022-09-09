// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#include "dnnl_onnx_tensor.h"

namespace onnxruntime {
namespace ort_dnnl {

OnnxTensor::OnnxTensor(const NodeArg* arg) {
  // because the passed in ort graph will be released after compile
  // need to save the type/shape in dnnl IR
  arg_type_ = arg->Type();
  arg_type_proto_ = ONNX_NAMESPACE::TypeProto::Create();
  arg_type_proto_->copy_from(arg->TypeAsProto());
}

const ONNX_NAMESPACE::TensorShapeProto* OnnxTensor::GetShapeInOnnxFormat() const {
  if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
    return nullptr;
  }

  if (arg_type_proto_->value_case() != ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType) {
    return nullptr;
  }
  auto& tensor_type = arg_type_proto_->tensor_type();
  if (tensor_type.has_shape()) {
    return &tensor_type.shape();
  }
  return nullptr;
}

dnnl::memory::dims OnnxTensor::GetShapeInDnnlFormat() const {
  if (arg_type_proto_ == nullptr || arg_type_ == nullptr) {
    return dnnl::memory::dims();
  }
  auto* shape_proto = GetShapeInOnnxFormat();
  // a shape without any information
  if (shape_proto == nullptr) {
    LOGS_DEFAULT(INFO) << "nullptr shape for " << arg_type_;
    return dnnl::memory::dims();
  }
  std::vector<int64_t> shape;
  const auto& dims = shape_proto->dim();
  for (const auto& dim : dims) {
    bool has_dim_value = dim.value_case() == dim.kDimValue;
    if (!has_dim_value) {
      LOGS_DEFAULT(INFO) << "Dynamic shape for " << arg_type_;
      shape.push_back(DNNL_RUNTIME_DIM_VAL);
    } else {
      shape.push_back(dim.dim_value());
    }
  }
  // make scaler as having dimension of 1
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  auto dnnl_dims = dnnl::memory::dims(shape);
  return dnnl_dims;
}

dnnl::memory::data_type OnnxTensor::GetDataTypeInDnnlFormat() const {
  if (arg_type_proto_ == nullptr) {
    ORT_THROW("Invoke New_DnnlTensor's arg_type_proto_ not initialized yet.");
  }
  auto data_type = arg_type_proto_->tensor_type().elem_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      return dnnl::memory::data_type::undef;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return dnnl::memory::data_type::f16;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return dnnl::memory::data_type::f32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      // OneDNN does not have support for tensors of int64_t so we just say
      // the tensor is int32_t and then use casting in the actual operator
      // to convert the dnnl::memory::data_handle to an int64_t*.  Care
      // must be taken that an int64_t tensor does not make it pass the
      // node capability check unless the operator is explicitly expecting
      // the int64_t
      return dnnl::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return dnnl::memory::data_type::s32;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return dnnl::memory::data_type::s8;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return dnnl::memory::data_type::u8;
      // Same here, we use u8 as the handler for bool
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return dnnl::memory::data_type::u8;
    default:
      ORT_THROW("Unsupported data type: ", data_type);
  }
}


}  // namespace ort_dnnl
}  // namespace onnxruntime