// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_node.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

namespace onnxruntime {
namespace ort_dnnl {

New_DnnlNode::New_DnnlNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs) {
  GetInfoFromOnnxNode(node);
  SetInputs(inputs);
  SetOutputs(outputs);
  primitive_name_ = primitive_utils::OrtOperatorToDnnlPrimitiveName(node->OpType());
}

New_DnnlNode::New_DnnlNode( const std::string& op_type, 
                            const std::string& name,
                            DnnlTensorPtrVector& inputs, 
                            DnnlTensorPtrVector& outputs) 
  : node_name_(name), op_type_(op_type), primitive_name_(primitive_utils::OrtOperatorToDnnlPrimitiveName(op_type)) {
  SetInputs(inputs);
  SetOutputs(outputs);
}

DnnlTensorPtr New_DnnlNode::GetInput(NodeIndex input_idx) {
  // Check for out-of-bounds/non-existent inputs
  if (input_idx >= inputs_->size()) {
    return empty_tensor;
  }
  // Get input
  return *(inputs_->at(input_idx));
}

DnnlTensorPtr New_DnnlNode::GetOutput(NodeIndex output_idx) {
  // Check for out-of-bounds/non-existent outputs
  if (output_idx >= outputs_->size()) {
    return empty_tensor;
  }
  // Get output
  return *(outputs_->at(output_idx));
}

void New_DnnlNode::SetInPlaceOutput(size_t input_idx, size_t output_idx) {
  // Check for out-of-bounds/non-existent outputs
  if ((output_idx < outputs_->size()) && (input_idx < inputs_->size())) {
    *outputs_->at(output_idx) = *inputs_->at(input_idx);
  } else {
    ORT_THROW("[oneDNN] Error: Tried to set a primitive with out of bounds index");
  }
}

bool New_DnnlNode::HasDynamicInputs() {
  // For each input
  for (auto input : *inputs_) {
    // If it has dynamic dims
    if ((*input)->HasDynamicDims()) {
      // Return true
      return true;
    }
  }
  // Else return false
  return false;
}

DnnlPrimitiveConfig New_DnnlNode::GeneratePrimitiveConfig() {
  auto config = DnnlPrimitiveConfig();
  config.src_primitive = PrimitiveName();
  config.src_out_shape = OutputDims();
  config.src_in_type = GetInput(0)->DataType();
  return config;
}

void New_DnnlNode::GetInfoFromOnnxNode(const Node* node) {
  // Copy Node basic characteristics
  op_type_ = node->OpType();
  attr_->insert(node->GetAttributes());
  since_version_ = node->SinceVersion();
  index_ = node->Index();
  node_name_ = node->Name();
}

}  // namespace ort_dnnl
}  // namespace onnxruntime