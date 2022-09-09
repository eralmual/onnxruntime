// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

// Class used to store info from the original ONNX tensor
struct OnnxTensor {
 public:
  OnnxTensor(const NodeArg* arg);
  // Get tensor data type
  dnnl::memory::data_type GetDataTypeInDnnlFormat() const;
  // Get onnx tensor shape
  const ONNX_NAMESPACE::TensorShapeProto* GetShapeInOnnxFormat() const;
  dnnl::memory::dims GetShapeInDnnlFormat() const;
 private:
  // Onnx tensor data type
  ONNX_NAMESPACE::DataType arg_type_ = nullptr;
  std::unique_ptr<ONNX_NAMESPACE::TypeProto> arg_type_proto_ = nullptr;
};



}  // namespace ort_dnnl
}  // namespace onnxruntime
