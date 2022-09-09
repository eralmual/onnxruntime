// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_graph_transformer.h"

// For debbuging purposes
#define DNNL_GRAPH_TRANSFORM_LOG 0

namespace onnxruntime::ort_dnnl {

New_DnnlGraphTransformer::New_DnnlGraphTransformer() : DnnlGraphBuilder() {
  // Enable logging if needed
  const std::string debug_log_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DEBUG_LOG");
  if (!debug_log_env.empty()) {
    debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
  }
  // Initialize the optimization sequence
  optim_seq_ = std::make_unique<DnnlOptimizedSequence>();
}

void New_DnnlGraphTransformer::TransformNodes(const onnxruntime::GraphViewer& onnx_graph) {
  ORT_UNUSED_PARAMETER(onnx_graph);
  // Get the graph size
  auto graph_size = nodes_.size();
  // Iterate over each node checking for post-op capability
  for (NodeIndex idx = 0; idx < graph_size; ++idx) {
    auto op_type = GetNode(idx)->OpType();
    // This can definetly be improved, just did it like this for now
    if (op_type == "MatMulInteger") {
      RemoveMatMulIntegerZp(onnx_graph, idx);
    }
  }
}

void New_DnnlGraphTransformer::PostOpOptimization() {
  // Get the graph size
  auto graph_size = nodes_.size();
  // Generate container 
#if DNNL_GRAPH_TRANSFORM_LOG
  printf("----------------------- Creating PostOps -----------------------\n");
#endif

  // Create the optimized sequence object 
  auto op_seq = std::make_shared<DnnlHyperNode>();
  // Set the source node and prepare it
  op_seq->src_node = GetNode(0);
  // Get the PostOpConfig
  auto post_op_config = op_seq->src_node->GeneratePrimitiveConfig();
 
  // Get auxiliary variables ready
  New_DnnlNode* last_fused = op_seq->src_node;

  // Iterate over each node checking for post-op capability
  for (NodeIndex idx = 0; idx < graph_size; ++idx) {
    // If there is a next node
    if ((idx + 1) < graph_size) {
      // Get next node
      auto next_node = GetNode(idx + 1);

      // If nodes are compatible then generate the post op and continue with iteration
      if (AreNodesPostOpCompatible(last_fused, next_node, &post_op_config)) {
#if DNNL_GRAPH_TRANSFORM_LOG
        printf("Adding %s_%i as %s post-op #%i \n", next_node->Name().c_str(), static_cast<int>(next_node->Index()),
               op_seq->src_node->OpType().c_str(), static_cast<int>(post_op_config.num_post_ops));
#endif
        // Some cast ops are not seen as post ops since we use primtive implicit casting
        if (next_node->OpType() != "Cast") {
          // Add node to the list;
          op_seq->post_op_info.push_back(DnnlPostOpInfo(next_node, post_op_config.input_idx));
          // Update post-op counter
          ++post_op_config.num_post_ops;
        }
        // Update last fused
        last_fused = next_node;

        continue;

        // If they are not compatible, then set the build_node as src_node and src_node to next_node
      } else {
#if DNNL_GRAPH_TRANSFORM_LOG
        printf("Storing sequence node %s\n", op_seq->src_node->Name().c_str());
#endif
        // Store current sequence
        optim_seq_->push_back(op_seq);
        // Build a new one
        op_seq = std::make_shared<DnnlHyperNode>();
        // Set the source node and prepare it
        op_seq->src_node = next_node;
        // Set the PostOpConfig
        post_op_config = op_seq->src_node->GeneratePrimitiveConfig();
      }
    } else {
#if DNNL_GRAPH_TRANSFORM_LOG
      printf("Storing sequence node %s\n", op_seq->src_node->Name().c_str());
#endif
      // If there are no more nodes then store current sequence
      optim_seq_->push_back(op_seq);
      op_seq.reset();
    }

    // If we have fusion then reset the tensor ptrs
    if (last_fused != optim_seq_->back()->src_node) {
      optim_seq_->back()->src_node->SetOutputs(last_fused->GetOutputs());
    }
   
    // Get ready for next it if any
    if (op_seq) {
      // Adjust last fused to point new src
      last_fused = op_seq->src_node;
    }
  }
}

void New_DnnlGraphTransformer::RemoveMatMulIntegerZp(const onnxruntime::GraphViewer& onnx_graph, NodeIndex index) {
  // Get the MatMulInteger node
  auto dnnl_node = GetNode(index);
  // The Zero point B index
  const auto zp_b_idx = 3;
  // Check if zero point B exists
  if (!(dnnl_node->InputCount() >= 4 && dnnl_node->GetInput(zp_b_idx)->Exists())) {
    return;
  }
  // Get the tensor from the onnx graph
  auto b_zero_point = dnnl_node->GetInput(zp_b_idx);
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  // Verify we found it
  if (!onnx_graph.GetInitializedTensor(b_zero_point->Name(), tensor_proto)) {
    return;
  }
  // If the tensor is empty return
  if (tensor_proto == nullptr) {
    return;
  }
  // Call the number of elements in the tensor
  const auto& dims = tensor_proto->dims();
  int num_elements = 1;
  for (int i = 0; i < tensor_proto->dims_size(); i++) {
    num_elements *= static_cast<int>(dims[i]);
  }

  // Check if b_zp is all zeros, assume data is s8 since only s8 weight is supported in onednn
  bool all_zero = true;
  std::vector<int8_t> unpacked_tensor;
  unpacked_tensor.resize(num_elements, 1);
  ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackTensor(
                      *tensor_proto, tensor_proto->has_raw_data() ?  tensor_proto->raw_data().data() : nullptr,
                      tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0, 
                      reinterpret_cast<int8_t*>(unpacked_tensor.data()), num_elements));
  for (const auto& val : unpacked_tensor) {
    if (val != 0) {
      all_zero = false;
      break;
    }
  }
  // If some elements are NOT zero then return since we can't remove it
  if (!all_zero) {
    return;
  }
  // Log removal
  if (debug_log_) {
    LOGS_DEFAULT(ERROR) << "Remove weight ZP of [" << dnnl_node->Name() << "]";
  }
  // Remove tensor from graph
  RemoveTensor(b_zero_point->Name());
}

int New_DnnlGraphTransformer::GetInputConnectionIndex(New_DnnlNode* src_node, New_DnnlNode* next_node) {
  for (int i = 0; i < src_node->OutputCount(); ++i) {
    // Get output from src
    auto src_output = src_node->GetOutput(i);
    for (int j = 0; j < next_node->InputCount(); ++j) {
      // Get input from next
      auto next_input = next_node->GetInput(j);
      // if src output[i] is == to next input[j] then means the nodes are conected
      if (src_output == next_input) {
        return j;
      }
    }
  }
  // If no connection found then -1
  return -1;
}

bool New_DnnlGraphTransformer::AreNodesPostOpCompatible(New_DnnlNode* src_node, New_DnnlNode* next_node,
                                                        DnnlPrimitiveConfig* post_op_config) {
  // Check if the nodes are comnnected
  post_op_config->input_idx = GetInputConnectionIndex(src_node, next_node);
  if (post_op_config->input_idx == -1) {
    return false;
  }
  // Check if node has more than one output
  auto output_tensor = src_node->GetOutput(0);
  if (src_node->OutputCount() > 1) {
    return false;
  }
  // If the node is consumed by more than one node
  if (output_tensor->NumConsumers() > 1) {
    return false;
  }
  // If src node output is part of the graph outputs
  if (output_tensor->IsGraphOutput()) {
    return false;
  }
  // Check if we have 32 or more post ops, then there is no more space
  if (post_op_config->num_post_ops >= 32) {
    return false;
  }
  // We alway asume that the so called next node is NOT a Hypernode
  // Check if next node's primitive allows for post-oping
  if (!next_node->IsPostOpCompatible(post_op_config)) {
    return false;
  }
  // If none of the above conditions happen, then we are compatible
  return true;
}

}  // namespace onnxruntime::ort_dnnl
