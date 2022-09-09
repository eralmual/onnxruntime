// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_graph.h"

// For debbuging purposes
#define DNNL_GRAPH_PRIMITIVE_CREATION 0

namespace onnxruntime::ort_dnnl {

New_DnnlGraph::New_DnnlGraph(std::string name, const GraphViewer& graph_viewer) : New_DnnlGraphTransformer() {
  // The graph is built in topological order
  BuildGraph(graph_viewer);

  // Remember input and and initialized tensors of the graph in order to set the memory later
  for (const auto* node_arg : graph_viewer.GetInputsIncludingInitializers()) {
    input_names_.insert(node_arg->Name());
    // Set as input to correctly set data handles
    GetTensor(node_arg->Name())->SetAsGraphInput();
  }

  for (const auto* node_arg : graph_viewer.GetOutputs()) {
    output_names_.insert(node_arg->Name());
    // Set the tensor as output to prevent inplace ops errors
    GetTensor(node_arg->Name())->SetAsGraphOutput();
  }

  // Apply transforms
  TransformNodes(graph_viewer);
  // Optimizes the execution stream
  PostOpOptimization();

  // Select the input node
  // Evaluate if its inputs are dynamic
  for (auto input : input_names_) {
    // If it has dynamic dims
    if (GetTensor(input)->HasDynamicDims()) {
      // Return true
      has_dynamic_inputs_ = true;
    }
  }
  // Store the graph name
  name_ = name;
  // Initialize the DnnlHandler
  dnnl_handler_ = std::make_unique<DnnlHandler>();
}

bool New_DnnlGraph::HasDynamicInputs() {
  return has_dynamic_inputs_;
}


void New_DnnlGraph::SetInputs(const std::unordered_map<std::string, New_OnnxTensorData>& inputs) {
  // Over here, we assign memory to our graph input tensors
  for (auto& input_tensor_name : input_names_) {
    auto input_tensor = GetTensor(input_tensor_name);
    // The tensor might have been deleted by fusion so verify it exists
    if (input_tensor->Exists()) {
      // Get the input dims
      dnnl::memory::dims dnnl_dims = inputs.at(input_tensor->Name()).tensor_info.shape;
      // If scalar we need to add dimensions since ORT({}) and DNNL({1}) representation of scalars differ
      if (dnnl_dims.size() == 0) {
        dnnl_dims.push_back(1);
        input_tensor->SetAsScalar();
      }
      // Create the input md
      auto input_md = dnnl::memory::desc(dnnl_dims, input_tensor->DataType(), FormatMap.at(dnnl_dims.size()));
      // Create the memory object in the input tensor
      input_tensor->SetMemory(
          std::make_unique<dnnl::memory>(input_md,
                                         dnnl_handler_->GetEngine(),
                                         inputs.at(input_tensor->Name()).buffer));
    }
  }
}

void New_DnnlGraph::GeneratePrimitives() {
  New_DnnlNode* src_node;
  DnnlPrimitiveConfig config;
  auto dnnl_engine = dnnl_handler_->GetEngine();
  // Iterate over each fused node
  for (auto& optimized_sequence : *optim_seq_) {
    // Get src and recalculate dims with defined input size
    src_node = optimized_sequence->src_node;
    src_node->CalculateOutputDims();
#if DNNL_GRAPH_PRIMITIVE_CREATION
    printf("Generating %s post-ops \n", optimized_sequence->src_node->Name().c_str());
#endif
    // Generate primitive config
    config = src_node->GeneratePrimitiveConfig();
    // Iterate over the fused nodes to generate all the post ops
    for (auto& post_op : optimized_sequence->post_op_info) {
#if DNNL_GRAPH_PRIMITIVE_CREATION
      printf("Generating %s as post-op primitive \n", post_op.post_op_node->Name().c_str());
#endif
      config.input_idx = post_op.fused_tensor_idx;
      post_op.post_op_node->GeneratePrimitive(config);
    }
    // Generate src primitive and append post ops
    auto primitives = src_node->GeneratePrimitive(config, dnnl_engine);
    // Insert primitives into the list
    primitives_.insert(primitives_.end(), primitives->begin(), primitives->end());
  }
}

inline void New_DnnlGraph::VerifyOutputs() {
  // ONNX Runtime expects the memory in plain format, so make sure every output has the correct layout
  for (auto& output_tensor_name : output_names_) {
    auto output_tensor = GetTensor(output_tensor_name);
    // Ideal memory desc
    auto plain_desc = dnnl::memory::desc(output_tensor->Dims(), output_tensor->DataType(),
                                         FormatMap.at(output_tensor->Dims().size()));
    // If we have either different md's or engines, reorder
    if (output_tensor->MemoryDesc() != plain_desc 
        || output_tensor->Memory().get_engine() != dnnl_handler_->GetCPUEngine()) {
      // Add the primitive to the list
      output_tensor->ReorderMemory(plain_desc, &primitives_);
    }
  }
}

void New_DnnlGraph::Compile(const std::unordered_map<std::string, New_OnnxTensorData>& inputs) {
  // If the key isn't empty means we already compiled and if the graph is static there is no 
  // reason to compile again
  if (!shape_key_.empty() && !HasDynamicInputs()) {
    return;
  }

  // Generate a key for the inputs
  std::string key;
  for (auto input : inputs) {
    for (auto dim : input.second.tensor_info.shape) {
      std::ostringstream o;
      o << dim;
      key += o.str();
      key += ",";
    }
    key += "|";
  }

  // If new key if different from shape_key_, update and recompile
  if (key != shape_key_) {
    shape_key_ = key;
  } else {
    return;
  }

  // Log compilation
  if (HasDynamicInputs()) {
    LOGS_DEFAULT(INFO) << "Dynamic Compile";
  } else {
    LOGS_DEFAULT(INFO) << "Static Compile";
  }

  // Reset tensor internal configuration
  for (auto& node : nodes_) {
    for (auto& in_tensor : *(node->GetInputs())) { (*in_tensor)->ResetTensorConfig(); }
    for (auto& out_tensor : *(node->GetOutputs())) { (*out_tensor)->ResetTensorConfig(); }
  }
  //Clean the primitive list
  primitives_.clear();

#if DNNL_ORT_GRAPH_LOG
  printf("----------------------- Compiling graph %s -----------------------\n", name_.c_str());
#endif

  // Assign memory to graph inputs
  SetInputs(inputs);

  // Generates the primitives for the graph
  GeneratePrimitives();

  // Make sure output tensors are in plain format
  VerifyOutputs();
}

void New_DnnlGraph::SetDataHandles(const std::unordered_map<std::string, New_OnnxTensorData>& inputs,
                                              const std::unordered_map<std::string, New_OnnxTensorData>& outputs) {

  auto dnnl_stream = dnnl_handler_->GetStream();

  for (auto& input : inputs) {
    if (Contains(input_names_, input.first) && GetTensor(input.first)->Exists()) {
      GetTensor(input.first)->SetMemoryDataHandle(input.second.buffer, dnnl_stream);
    }
  }

  /*printf("----------------------- Tensors at handle setting time -----------------------\n");
  for (auto tensor_name : input_names_) {
    auto tensor = GetTensor(tensor_name);
    if (tensor->Name() != "") {
      printf("Input %s = %p\n", tensor_name.c_str(), tensor->Memory().get_data_handle());
      printf("%s tensor ptr = %p\n", tensor->Name().c_str(), tensor);
      auto mem = static_cast<float*>(tensor->Memory().get_data_handle());
      auto limit = 4;
      printf("%s = {", tensor_name.c_str());
      for (int i = 0; i < limit; ++i) {
        printf("%f, ", mem[i]);
      }
      printf("}\n");
    }
  }*/

  for (auto& output : outputs) {
    if (Contains(output_names_, output.first)) {
      GetTensor(output.first)->SetMemoryDataHandle(output.second.buffer, dnnl_stream);
      dnnl_stream.wait();
    }
  }

  /*for (auto tensor_name : output_names_) {
    auto tensor = GetTensor(tensor_name);
    if (tensor->Name() != "") {
      printf("%s tensor ptr = %p\n", tensor->Name().c_str(), tensor);
    }
  }*/
  //printf("-----------------------------------------------------------------------------\n");
}

onnxruntime::common::Status New_DnnlGraph::Predict() {
  // Get stream
  auto dnnl_stream = dnnl_handler_->GetStream();

  // Run inference
  for (auto& primitive : primitives_) {
    primitive.first.execute(dnnl_stream, primitive.second);
    dnnl_stream.wait();
    /*for (auto& io : primitive.second) {
      auto mem = static_cast<float*>(io.second.get_data_handle());
      auto limit = 10;
      printf("tensor map #%i = {", io.first);
      for (int i = 0; i < limit; ++i) {
        printf("%f, ", mem[i]);
      }
      printf("}\n\n");
    }*/
  }

  /*for (auto tensor_name : output_names_) {
    auto tensor = GetTensor(tensor_name);
    auto mem = static_cast<int32_t*>(tensor->Memory().get_data_handle());
    auto limit = (tensor->Dims()[0] * tensor->Dims()[1]) < 10 ? tensor->Dims().size() : 10;
    printf("%s = {", tensor_name.c_str());
    for (int i = 0; i < limit; ++i) {
      printf("%i, ", static_cast<int>(mem[i]));
    }
    printf("}\n\n");
  }*/

  return Status::OK();
}


}  // namespace onnxruntime::ort_dnnl
