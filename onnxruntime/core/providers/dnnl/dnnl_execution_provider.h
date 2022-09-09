// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <list>
#include <memory.h>

#include "core/platform/ort_mutex.h"
#include "dnnl_op_manager.h"
#include "core/providers/dnnl/graph/dnnl_graph.h"

namespace onnxruntime {
//static long long build_time = 0;
//static long long comp_time = 0;
//static long long pred_time = 0;

// Information needed to construct DNNL execution providers.
struct DNNLExecutionProviderInfo {
  bool create_arena{true};

  explicit DNNLExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}
  DNNLExecutionProviderInfo() = default;
};

// Logical device representation.
class DNNLExecutionProvider : public IExecutionProvider {
 public:
  explicit DNNLExecutionProvider(const DNNLExecutionProviderInfo& info);
  virtual ~DNNLExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
   // NEW STUFF
  /* Used to store the Graph and its I/O.
  To identify each object we use the graph name provided by ORT, with this
  we get the DnnlGraph, and the name of the input and output tensors also provided by ORT
  with this we can then use the I/O names to set the data handles into the graph, when available.
  */ 
  std::unordered_map<std::string, ort_dnnl::DnnlGraphInfo> graphs_;
  // DnnlOpManager contains information about supported Dnnl Operators
  DnnlOpManager opManager_;
  std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer) const;
  // dump subgraphs to onnx format for debugging purpose
  bool dump_subgraphs_ = false;
  bool debug_log_ = false;
  //enable fusion by default
  bool enable_fusion_ = true;
};

}  // namespace onnxruntime
