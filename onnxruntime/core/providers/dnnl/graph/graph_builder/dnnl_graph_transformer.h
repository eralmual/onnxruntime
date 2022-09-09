// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <string>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/dnnl/graph/graph_builder/dnnl_graph_builder.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

namespace onnxruntime::ort_dnnl {


// Used to store post-op configuration info
struct DnnlPostOpInfo {
  // Node we are going to add as a post-op
  New_DnnlNode* post_op_node;
  // Input tensor that is bign fused into post-op
  //DnnlTensorPtr fused_tensor;
  // post_op_node input index of the tensor that is output of the src node
  int fused_tensor_idx;
  // Define small constructor for easier instantiation
  DnnlPostOpInfo(New_DnnlNode* post_op_node_ptr /*, New_DnnlTensor* fused_tensor_ptr*/, int fused_tensor_idx_val)
      : post_op_node(post_op_node_ptr)/*, fused_tensor(fused_tensor_ptr)*/, fused_tensor_idx(fused_tensor_idx_val){};
};

// Used to store optimized subgraphs
struct DnnlHyperNode {
  New_DnnlNode* src_node;
  DnnlPrimitiveConfig config;
  std::vector<DnnlPostOpInfo> post_op_info;
  // Define small constructor for easier instantiation
  DnnlHyperNode() = default;
};

using DnnlOptimizedSequence = std::vector<std::shared_ptr<DnnlHyperNode>>;

class New_DnnlGraphTransformer : public DnnlGraphBuilder {
 public:
  New_DnnlGraphTransformer();
  
 protected:
  // Debug flag
  bool debug_log_ = false;
  // Secuence of fused primitives
  std::unique_ptr<DnnlOptimizedSequence> optim_seq_;

  // Looks for specific fusion patterns and modifies the graph inplace, takes one iteration over the graph
  void TransformNodes(const onnxruntime::GraphViewer& onnx_graph);  // make private later

  // Does the post-op fusion and generates a list of primitives
  void PostOpOptimization();

 private:
  // Fusion functions
  void RemoveMatMulIntegerZp(const onnxruntime::GraphViewer& onnx_graph, NodeIndex index);
  
  // Get index of a tensor that connects two nodes, return -1 if there is no connection
  int GetInputConnectionIndex(New_DnnlNode* src_node, New_DnnlNode* next_node);
  // Evaluate if two nodes can be fused
  bool AreNodesPostOpCompatible(New_DnnlNode* src_node, 
                                New_DnnlNode* next_node,
                                DnnlPrimitiveConfig* post_op_config);
};
}  // namespace onnxruntime::ort_dnnl