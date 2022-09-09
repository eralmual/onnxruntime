// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/providers/dnnl/graph/primal_nodes/dnnl_binary_node.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_eltwise_node.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_matmul_node.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_reorder_node.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_layernorm_node.h"

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime::ort_dnnl {

using UniqueVectorOfUniqueNodes = std::unique_ptr<std::vector<UniqueDnnlNodePtr>>;

class DnnlGraphBuilder {

 public:
  DnnlGraphBuilder() = default;

  New_DnnlNode* GetNode(NodeIndex node_index) { return nodes_.at(node_index).get(); }
  std::vector<New_DnnlNode*> GetNodes();
  New_DnnlTensor* GetTensor(const std::string& name);
  // Graph info setters
  void SetNode(UniqueDnnlNodePtr new_node) { nodes_.emplace_back(std::move(new_node)); }
  void SetNodes(std::vector<UniqueDnnlNodePtr>& new_nodes);
  void RemoveNode(size_t node_index) { nodes_.at(node_index).release(); }
  void RemoveTensor(std::string name) { *tensors_[name] = *tensors_[empty_tensor_name]; }
 
 protected:
  // List of nodes that compose the graph
  std::vector<UniqueDnnlNodePtr> nodes_;
  // List of references to tensors, required to support inplace and memory ops
  std::unordered_map<std::string, UniqueSharedDnnlTensorPtr> tensors_;
  // Used to fill for empty tensors
  inline static const std::string empty_tensor_name = "Empty_tensor";

  // Builds the node operator and returns the names of the outputs in the ONNX specified order
  void BuildOperator(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);
  // Build the dnnl graph as a list of nodes
  void BuildGraph(const GraphViewer& graph_viewer);

private:
  // QAttention related functions
  void BuildQAttention(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);

  // Skip layer norm related functions
  void BuildSkipLayerNorm(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);
  float GetLayerNormEpsilon(const Node* node);

  // Bias GeLU related functions
  void BuildBiasGelu(const Node* node, DnnlTensorPtrVector& inputs, DnnlTensorPtrVector& outputs);
};

}  // namespace onnxruntime::ort_dnnl