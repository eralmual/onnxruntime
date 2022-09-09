// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_graph_builder.h"

#include "dnnl.hpp"

#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

// For debbuging purposes
#define DNNL_GRAPH_BUILD_LOG 0

namespace onnxruntime::ort_dnnl {

void DnnlGraphBuilder::BuildOperator(const Node* node,
                                        DnnlTensorPtrVector& inputs,
                                        DnnlTensorPtrVector& outputs) {
  // Get the node op type
  auto op_type = node->OpType();

  // Consume all inputs
  for (auto& input : *inputs) {
    (*input)->Consume();
  }

  // Check if we need hyper nodes or is a simple op with 1:1 mapping
  if (primitive_utils::IsSimpleOperator(op_type)) {
    // Get the pritive type
    auto primitive_type = primitive_utils::OrtOperatorToDnnlPrimitiveName(op_type);
    // Generate the primal node
    if (primitive_type == "Binary") {
      nodes_.emplace_back(std::make_unique<DnnlBinaryNode>(node, inputs, outputs));
    } else if (primitive_type == "Eltwise") {
      nodes_.emplace_back(std::make_unique<DnnlEltwiseNode>(node, inputs, outputs));
    } else if (primitive_type == "MatMul") {
      nodes_.emplace_back(std::make_unique<DnnlMatMulNode>(node, inputs, outputs));
    } else if (primitive_type == "Reorder") {
      nodes_.emplace_back(std::make_unique<DnnlReorderNode>(node, inputs, outputs));
    } else if (primitive_type == "LayerNormalization") {
      nodes_.emplace_back(std::make_unique<DnnlLayerNormNode>(node, inputs, outputs));
    } else {
      throw std::invalid_argument("Dnnl primitive not found");
    }
  } else {
    if (op_type == "QAttention") {
      BuildQAttention(node, inputs, outputs);
    } else if (op_type == "SkipLayerNormalization") {
      BuildSkipLayerNorm(node, inputs, outputs);
    } else if (op_type == "BiasGelu") {
      BuildBiasGelu(node, inputs, outputs);
    } else {
      ORT_THROW("Trying to build unknown hypernode: ", op_type);
    }
  }
}

void DnnlGraphBuilder::BuildGraph(const GraphViewer& graph_viewer) {
#if DNNL_GRAPH_BUILD_LOG
  printf("----------------------- Building graph %s -----------------------\n", graph_viewer.Name().c_str());
#endif
  //  Get all nodes of the onnx graph in topological order
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  // Get number of nodes
  auto num_nodes = node_indices.size();
  // Reserve memory for the minimum amout of nodes
  nodes_.reserve(num_nodes);
  // Create an empty tensor
  tensors_.insert({empty_tensor_name, dnnl_make_tensor()});
  for (size_t i = 0; i < num_nodes; i++) {
    // Get the ORT node
    auto node = graph_viewer.GetNode(node_indices[i]);
    // Generate container for inputs and outputs
    auto inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    auto outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
#if DNNL_GRAPH_BUILD_LOG
    printf("Node: %s_%i\n", node->Name().c_str(), static_cast<int>(node->Index()));
#endif
    // Iterate over the onnx node inputs
    inputs->reserve(node->InputDefs().size());
    for (auto input : node->InputDefs()) {
#if DNNL_GRAPH_BUILD_LOG
      printf("\tInput: %s\n", input->Name().c_str());
#endif
      // If the input exists and has a valid name
      if (input && input->Exists() && input->Name() != "") {
        // If there is no input tensor registered with this name
        if (!Contains(tensors_, input->Name())) {
          // Store tensor info and if it's a constant initializer, use an extra level of indirection for inplace ops
          tensors_.insert({input->Name(), dnnl_make_tensor(input, graph_viewer.IsConstantInitializer(input->Name(), true))});
        }
        // Set the tensor as the input of the DnnlNode
        inputs->emplace_back(tensors_.at(input->Name()).get());
      } else {
        // Non existent or invalid input, store as empty tensor
        inputs->emplace_back(tensors_.at("Empty_tensor").get());
      }
    }

    // Iterate over the onnx node outputs
    outputs->reserve(node->OutputDefs().size());
    for (auto output : node->OutputDefs()) {
#if DNNL_GRAPH_BUILD_LOG
      printf("\tOutput: %s\n", output->Name().c_str());
#endif
      // If the outputs exists and has a valid name
      if (output && output->Exists() && output->Name() != "") {
        // If there is no outputs tensor with this name
        if (!Contains(tensors_, output->Name())) {
          // Store tensor info and if it's a constant initializer, use an extra level of indirection for inplace ops
          tensors_.insert({output->Name(), dnnl_make_tensor(output, false)});
        }
        // Set the tensor as the output of the DnnlNode
        outputs->emplace_back(tensors_.at(output->Name()).get());
      } else {
        // Non existent or invalid output, store as empty tensor
        outputs->emplace_back(tensors_.at("Empty_tensor").get());
      }
    }
    // Build the operator as a secuence of dnnl nodes
    BuildOperator(node, inputs, outputs);
  }
}

std::vector<New_DnnlNode*> DnnlGraphBuilder::GetNodes() {
  std::vector<New_DnnlNode*> result;
  for (auto& node : nodes_) {
    if (node.get()) {
      result.push_back(node.get());
    }
  }
  return result;
}

New_DnnlTensor* DnnlGraphBuilder::GetTensor(const std::string& name) {
  if (Contains(tensors_, name)) {
    return (*tensors_.at(name)).get();
  } else {
    ORT_THROW("Could not find DnnlTensor named ", name);
  }
}

void DnnlGraphBuilder::SetNodes(std::vector<UniqueDnnlNodePtr>& new_nodes) {
  for (size_t i = 0; i < new_nodes.size(); ++i) {
    nodes_.emplace_back(std::move(new_nodes.at(i)));
  }
}

}  // namespace onnxruntime::ort_dnnl