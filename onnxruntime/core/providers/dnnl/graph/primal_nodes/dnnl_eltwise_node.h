// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlEltwiseNode : public New_DnnlNode {
 public:
  enum InputTensors : int {
    IN_X = 0,  // Required every op
    IN_Y = 1,  // Required by Pow operator
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlEltwiseNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(node, inputs, outputs), algorithm_(primitive_utils::OrtOperatorToDnnlAlgorithm(OpType())) 
        { CalculateOutputDims(); }

  DnnlEltwiseNode(const std::string& op_type, const std::string& name,
                  DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(op_type, name, inputs, outputs), algorithm_(primitive_utils::OrtOperatorToDnnlAlgorithm(op_type))
        { CalculateOutputDims(); }

  std::unique_ptr<std::vector<DnnlPrimitive>> GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                    const dnnl::engine& dnnl_engine) override;
  void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) override;
  bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) override;
  void BuildInPlace(DnnlPrimitiveConfig* post_op_config) override;

  void CalculateOutputDims() override;

 private:
  // Binary primitive specific attributes
  dnnl::algorithm algorithm_;

  // Get alpha value
  float GetAlpha();
  // Get beta value
  float GetBeta();
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
