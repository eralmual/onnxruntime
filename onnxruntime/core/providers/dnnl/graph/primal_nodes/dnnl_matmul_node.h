// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlMatMulNode : public New_DnnlNode {
 public:
  enum InputTensors : int {
    IN_A = 0,             // Required for MatMul and MatMulInteger
    IN_B = 1,             // Required for MatMul and MatMulInteger
    IN_A_ZERO_POINT = 2,  // Optional for MatMulInteger
    IN_B_ZERO_POINT = 3,  // Optional for MatMulInteger
    IN_OUTPUT_SCALE = 4,  // Optional, adds support for output scales for some ops and optimizations
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlMatMulNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(node, inputs, outputs) { CalculateOutputDims(); }

  DnnlMatMulNode(const std::string& op_type, const std::string& name, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(op_type, name, inputs, outputs) { CalculateOutputDims(); }

  std::unique_ptr<std::vector<DnnlPrimitive>> GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                    const dnnl::engine& dnnl_engine) override;
  void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) override;
  // MatMulInteger is not compatible as a post op
  bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) override;
  // MatMulInteger is not inplace compatible due to shapes
  void BuildInPlace(DnnlPrimitiveConfig* post_op_config) override;

  void CalculateOutputDims() override;

 private:

  // Cast zero point to s32
  inline void CastZeroPoint( New_DnnlTensor& zero_point, 
                              std::vector<DnnlPrimitive>* prim_list);
  // Checks if zero points are provided and adds them
  inline void AddZeroPoint(int tensor_idx, dnnl::primitive_attr& prim_attr, DnnlPrimitiveConfig& post_op_config);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
