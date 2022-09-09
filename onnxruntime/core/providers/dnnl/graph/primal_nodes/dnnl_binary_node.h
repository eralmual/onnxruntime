// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/shared_library/provider_api.h"


namespace onnxruntime {
namespace ort_dnnl {

class DnnlBinaryNode : public New_DnnlNode {
 public:

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

  DnnlBinaryNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs) 
    : New_DnnlNode(node, inputs, outputs) { CalculateOutputDims(); }

  DnnlBinaryNode(const std::string& op_type, const std::string& name,
                DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs) 
    : New_DnnlNode(op_type, name, inputs, outputs) { CalculateOutputDims(); }
  
  std::unique_ptr<std::vector<DnnlPrimitive>> GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                    const dnnl::engine& dnnl_engine) override;
  void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) override;
  bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) override;
  void BuildInPlace(DnnlPrimitiveConfig* post_op_config) override;

  void CalculateOutputDims() override;

 private:
  bool is_inplace_ = false;

  /* Detects if the input can used optimized broadcasting for post-ops, information extracted from 
  https://github.com/oneapi-src/oneDNN/blob/62c7bc5f5c73133f746868663c29f45bb0492b64/src/cpu/x64/jit_uni_binary.cpp#L28
  https://github.com/oneapi-src/oneDNN/blob/master/src/common/broadcast_strategy.hpp#L31
  * */
  bool IsBroadcastPostOpOptimized(dnnl::memory::dims& src_dims);

};

}  // namespace ort_dnnl
}  // namespace onnxruntime
