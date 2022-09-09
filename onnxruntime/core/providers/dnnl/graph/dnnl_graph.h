// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#pragma once

#include "dnnl.hpp"

#include <map>
#include <vector>
#include <string>
#include <limits>

#include "core/platform/ort_mutex.h"
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/dnnl/graph/dnnl_node.h"
#include "core/providers/dnnl/dnnl_handler/dnnl_handler.h"
#include "core/providers/dnnl/graph/graph_builder/dnnl_graph_transformer.h"
#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"


namespace onnxruntime::ort_dnnl {

// Reused these structures to minimize code changes in dnnl_ep.cc but we can sort this out in the future
// borrow from coreml ep's data structures to organize data handle, shape and data type
struct New_OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape;
};

struct New_OnnxTensorData {
  New_OnnxTensorInfo tensor_info;
  void* buffer{nullptr};
};

using TensorInputControl = std::unordered_map<std::string, std::vector<std::pair<New_DnnlNode*, size_t>>>;
using Name2TensorMap = std::unordered_map<std::string, std::shared_ptr<New_DnnlTensor>>;
using TensorControl = std::pair<Name2TensorMap, TensorInputControl>;

class New_DnnlGraph : public New_DnnlGraphTransformer {
 public:
  New_DnnlGraph(std::string name, const GraphViewer& graph_viewer);
  // Graph info getters
  inline std::string Name()   { return name_; }
  inline OrtMutex& GetMutex() { return mutex_; }
  void SetDataHandles(const std::unordered_map<std::string, New_OnnxTensorData>& inputs,
                                 const std::unordered_map<std::string, New_OnnxTensorData>& outputs);

  
  // Generates the primitives needed for execution and creates the dnnl::memory as needed
  void Compile(const std::unordered_map<std::string, New_OnnxTensorData>& inputs);
  // Run the graph
  onnxruntime::common::Status Predict();

  // Check whether the subgraph is dynamic
  bool HasDynamicInputs();  
  
 private:
  // List of inputs and initialized tensors of the graph
  std::unordered_set<std::string> input_names_;
  // List of output tensors of the graph
  std::unordered_set<std::string> output_names_;
  // Vector of primitives and its inputs
  std::vector<DnnlPrimitive> primitives_;
  // Graph name
  std::string name_;
  // Mutex
  OrtMutex mutex_;
  
  // Used to signal the graph contains dynamic inputs
  bool has_dynamic_inputs_ = false;
  // Used in compilation to detect changes in input size
  std::string shape_key_;
  // Helper for dnnl related functions
  std::unique_ptr<DnnlHandler> dnnl_handler_;

  // Resolves the graph using dnnl primitives, populates the primitives_ vector
  void SetInputs(const std::unordered_map<std::string, New_OnnxTensorData>& inputs);
  // Uses the existing nodes to generate the necessary primitives
  void GeneratePrimitives();
  // Make sure the output  memories are in the correct layout and on the cpu engine
  void VerifyOutputs();
};
// Used for storage on the dnnl ep
struct DnnlGraphInfo {
  std::unique_ptr<New_DnnlGraph> graph;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

}  // namespace onnxruntime::ort_dnnl
