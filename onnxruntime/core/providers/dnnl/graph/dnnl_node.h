// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_tensor.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_config.h"

namespace onnxruntime::ort_dnnl {

class New_DnnlNode {
 public:
  // Constructors
  // Constructor used for simple operators, cases where op maps to oneDNN primitive
  New_DnnlNode(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);
  // Constructor used for artificial node creation, I/O must be manually defined later
  New_DnnlNode(const std::string& op_type, const std::string& name, 
              DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);
  New_DnnlNode() = default;

  // Node basic characteristics getters
  inline const std::string& Name()    { return node_name_; }
  inline const NodeIndex& Index()     { return index_; }
  inline const std::string& OpType()  { return op_type_; }
  inline int SinceVersion()           { return since_version_; }
  bool HasDynamicInputs();

  // I/O and Attributes manipulation
  inline void SetInput(DnnlTensorPtr* input_tensor)         { inputs_->emplace_back(input_tensor); }
  inline void SetOutput(DnnlTensorPtr* output_tensor)       { outputs_->emplace_back(output_tensor); }
  inline void SetInputs(const DnnlTensorPtrVector& inputs)   { inputs_ = inputs; }
  inline void SetOutputs(const DnnlTensorPtrVector& outputs) { outputs_ = outputs; }
  DnnlTensorPtr GetInput(NodeIndex input_idx);
  DnnlTensorPtr GetOutput(NodeIndex output_idx);
  inline NodeAttributes* GetAttributes() { return attr_.get(); }
  inline const DnnlTensorPtrVector& GetInputs()    { return inputs_; }
  inline const DnnlTensorPtrVector& GetOutputs()   { return outputs_; }
  inline size_t InputCount()  { return inputs_->size(); }
  inline size_t OutputCount() { return outputs_->size(); }
  // Changes the tensor on a given output index
  void SetInPlaceOutput(size_t input_idx, size_t output_idx);

  // Primitive related
  // Generate the standalone primitive
  virtual DnnlPrimitiveVector GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                const dnnl::engine& dnnl_engine) = 0;
  // Generate the primitive as a post op
  virtual void GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) = 0;
  // Generate a primitive config object based of this node
  DnnlPrimitiveConfig GeneratePrimitiveConfig();
  // Evaluates if the primitive is supports as a post-op
  virtual bool IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) = 0;
  // Sets the primitive to be done in place if posible
  virtual void BuildInPlace(DnnlPrimitiveConfig* post_op_config) = 0;

  // Attributes getters
  // Returns the primitive name
  inline const std::string& PrimitiveName() { return primitive_name_; }
  inline dnnl::memory::dims OutputDims() { return *output_dims_; }

  // Do internal operations
  // Used to calculate output dims
  virtual void CalculateOutputDims() = 0;

  // Allows for proper destruction of Derived clases
  virtual ~New_DnnlNode() = default;

 private:
  // Node information 
  std::string node_name_;
  NodeIndex index_ = std::numeric_limits<NodeIndex>::max();   
  std::string op_type_ = "";
  int since_version_ = -1;
  // Operator requirements
  DnnlTensorPtrVector inputs_;
  DnnlTensorPtrVector outputs_;
  std::unique_ptr<NodeAttributes> attr_ = NodeAttributes::Create();

  // Private functions
  void GetInfoFromOnnxNode(const Node* node);

 protected:
  // Primitive requirements
  std::string primitive_name_;
  std::unique_ptr<dnnl::memory::dims> output_dims_;
};

// Add this for redability
using UniqueDnnlNodePtr = std::unique_ptr<New_DnnlNode>;

}  // namespace onnxruntime::ort_dnnl