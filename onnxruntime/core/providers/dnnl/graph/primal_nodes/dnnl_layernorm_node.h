// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlLayerNormNode : public New_DnnlNode {
 public:
  enum InputTensors : int {
    IN_X = 0,
    IN_SCALE = 1,
    IN_B = 2            // Bias (Optional)
  };

  enum OutputTensors : int {
    OUT_Y = 0,
    OUT_MEAN = 1,       // Optional
    OUT_INV_STDEV = 2   // Optional
  };

  DnnlLayerNormNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(node, inputs, outputs) { CalculateOutputDims(); }

  DnnlLayerNormNode(const std::string& op_type, const std::string& name, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs,
                    float epsilon)
      : New_DnnlNode(op_type, name, inputs, outputs), epsilon_(epsilon) { CalculateOutputDims(); }

  std::unique_ptr<std::vector<DnnlPrimitive>> GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                                      const dnnl::engine& dnnl_engine) override;
  void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) override;
  bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) override;
  void BuildInPlace(DnnlPrimitiveConfig* post_op_config) override;

  void CalculateOutputDims() override;

 private:
  bool is_inplace_ = false;
  float epsilon_ = -1;
  void GetEpsilon();
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
