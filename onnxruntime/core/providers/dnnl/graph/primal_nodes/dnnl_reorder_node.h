// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlReorderNode : public New_DnnlNode {
 public:
  enum InputTensors : int {
    IN_INPUT = 0,   // Required for Cast op
    IN_DATA = 0,    // Required for Unsqueeze op
    IN_AXES = 1,    // Required for Unsqueeze op
  };

  enum OutputTensors : int {
    OUT_OUTPUT = 0
  };

  DnnlReorderNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(node, inputs, outputs) { CalculateOutputDims(); }

  DnnlReorderNode(const std::string& op_type, const std::string& name, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs)
      : New_DnnlNode(op_type, name, inputs, outputs) { CalculateOutputDims(); }

  std::unique_ptr<std::vector<DnnlPrimitive>> GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                    const dnnl::engine& dnnl_engine) override;
  void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) override;
  bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) override;
  void BuildInPlace(DnnlPrimitiveConfig* post_op_config) override;

  void CalculateOutputDims() override;

 private:

  // Some ops require pivoting memory
  std::vector<std::unique_ptr<dnnl::memory>> aux_mem_;

  // Check if cast operator is post op compatible
  inline bool IsCastPostOpCompatble(DnnlPrimitiveConfig* post_op_config);
  inline void GenerateCast(dnnl::memory::desc& dst_md);
  inline dnnl::memory::data_type GetTo();

  // Unsueeze operator
  inline void GenerateUnsqueeze(const dnnl::engine& dnnl_engine);
  std::vector<int64_t> DnnlReorderNode::GetAxes();
};

}  // namespace ort_dnnl
}  // namespace onnxruntime
