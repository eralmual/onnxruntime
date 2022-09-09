// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include "core/providers/shared_library/provider_api.h"
#include <unordered_set>
#include "dnnl_execution_provider.h"
#include "dnnl_fwd.h"
#include "dnnl_node_capability.h"

//#include <iostream>

#include <iomanip>
#include <fstream>
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {

constexpr const char* DNNL = "Dnnl";
constexpr const char* DNNL_CPU = "DnnlCpu";

DNNLExecutionProvider::DNNLExecutionProvider(const DNNLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kDnnlExecutionProvider, true} {

  InitProviderOrtApi();

  AllocatorCreationInfo default_memory_info(
      {[](int) {
        return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(DNNL, OrtAllocatorType::OrtDeviceAllocator));
      }},
      0, info.create_arena);

  AllocatorCreationInfo cpu_memory_info(
      {[](int) {
        return onnxruntime::CreateCPUAllocator(OrtMemoryInfo(DNNL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0,
                                                             OrtMemTypeCPUOutput));
      }},
      0, info.create_arena);

  InsertAllocator(CreateAllocator(default_memory_info));
  InsertAllocator(CreateAllocator(cpu_memory_info));

  //debug env variable to dump subgraphs
  const std::string dump_subgraphs_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DUMP_SUBGRAPHS");
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
  }

  const std::string debug_log_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_DEBUG_LOG");
  if (!debug_log_env.empty()) {
    debug_log_ = (std::stoi(debug_log_env) == 0 ? false : true);
  }

  const std::string fusion_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_ENABLE_FUSION");
  if (!fusion_env.empty()) {
    enable_fusion_ = (std::stoi(fusion_env) == 0 ? false : true);
  }
}  // namespace onnxruntime

DNNLExecutionProvider::~DNNLExecutionProvider() {
}

std::vector<std::vector<NodeIndex>> DNNLExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer) const {
  std::vector<std::vector<size_t>> supported_node_vecs;
  std::vector<size_t> supported_node_vec;

  std::unordered_map<std::string,int> all_nodes_count;
  std::unordered_map<std::string,int> supported_nodes_count;

  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));

    bool supported = opManager_.IsNodeSupported(node, graph_viewer);

    //update count
    if(debug_log_){
      auto node_optype_ver = node->OpType() + "_" + std::to_string(node->SinceVersion());
      all_nodes_count[node_optype_ver]++;
      if (supported) {
        supported_nodes_count[node_optype_ver]++;
      }

      LOGS_DEFAULT(ERROR) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    }

    if (supported) {
      supported_node_vec.push_back(node_idx);
    } else {
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }

  //collect statistics and report
  if (debug_log_) {
    int all_counts = 0;
    int support_counts = 0;
    for(auto e: all_nodes_count){
      auto optype_ver = e.first;
      auto all_count = e.second;
      auto support_count = supported_nodes_count[optype_ver];
      all_counts += all_count;
      support_counts += support_count;
      LOGS_DEFAULT(ERROR) << "Operator type: [" << optype_ver << "] coverage: " << support_count << ":" << all_count << " percentage: " << (float)support_count / (float)all_count;
    }
    LOGS_DEFAULT(ERROR) << "Total coverge: " << support_counts << ":" << all_counts
                          << " percentage: " << (float)support_counts / (float)all_counts;
  }


  return supported_node_vecs;
}

std::vector<std::unique_ptr<ComputeCapability>> DNNLExecutionProvider::GetCapability(
    const GraphViewer& graph_viewer,
    const IKernelLookup& /*kernel_lookup*/) const {
  //follow from coreml ep's Getcapability

  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto node_groups = GetSupportedNodes(graph_viewer);

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  size_t num_of_supported_nodes = 0;
  for (const auto& group : node_groups) {
    if (group.empty())
      continue;

    num_of_supported_nodes += group.size();

    if (debug_log_) {
      LOGS_DEFAULT(ERROR) << "DNNLExecutionProvider::GetCapability, current supported node group size: "
                          << group.size();
    }

    std::unordered_set<NodeIndex> node_set;
    node_set.reserve(group.size());
    for (const auto& index : group) {
      node_set.insert(index);
    }

    auto sub_graph = onnxruntime::IndexedSubGraph::Create();
    std::unordered_set<const NodeArg*> node_outputs;
    std::unordered_set<const NodeArg*> subgraph_inputs;
    std::unordered_set<const NodeArg*> subgraph_outputs;
    std::vector<const NodeArg*> ordered_subgraph_inputs;
    std::vector<const NodeArg*> ordered_subgraph_outputs;

    for (const auto& index : group) {
      sub_graph->Nodes().push_back(index);
      const auto* node = graph_viewer.GetNode(index);

      for (const auto* input : node->InputDefs()) {
        // if the node input was not produced by this subgraph, add it to the subgraph inputs.
        if (!input->Exists()) {
          continue;
        }
        if (node_outputs.count(input) == 0) {
          if (subgraph_inputs.count(input) == 0) {
            subgraph_inputs.insert(input);
            ordered_subgraph_inputs.push_back(input);
          }
        }
      }

      const auto& output_defs = node->OutputDefs();
      for (const auto* output_def : output_defs) {
        if (!output_def->Exists()) {
          continue;
        }
        node_outputs.insert(output_def);
        // if output is overall graph output we need to produce it.
        if (graph_outputs.count(output_def) != 0) {
          ordered_subgraph_outputs.push_back(output_def);
        }
      }

      // if output connects to a node not in this subgraph we need to produce it
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        if (node_set.count(it->GetNode().Index()) == 0) {
          const auto* output_def = output_defs[it->GetSrcArgIndex()];
          if (subgraph_outputs.count(output_def) == 0 && graph_outputs.count(output_def) == 0) {
            subgraph_outputs.insert(output_def);
            ordered_subgraph_outputs.push_back(output_def);
          }
        }
      }
    }

    // Assign inputs and outputs to subgraph's meta_def
    HashValue model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    auto meta_def = ::onnxruntime::IndexedSubGraph_MetaDef::Create();
    meta_def->name() = "DNNL_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    meta_def->domain() = kMSDomain;
    meta_def->since_version() = 1;
    meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;

    for (const auto& input : ordered_subgraph_inputs) {
      meta_def->inputs().push_back(input->Name());
    }

    for (const auto& output : ordered_subgraph_outputs) {
      meta_def->outputs().push_back(output->Name());
    }

    sub_graph->SetMetaDef(std::move(meta_def));

    result.push_back(ComputeCapability::Create(std::move(sub_graph)));
  }

  if (debug_log_) {
    float percent_dnnl = 100.0f * (static_cast<float>(num_of_supported_nodes) / static_cast<float>(graph_viewer.NumberOfNodes()));
    LOGS_DEFAULT(ERROR) << "DNNLExecutionProvider::GetCapability,"
                        << " number of partitions supported by DNNL: " << result.size()
                        << " number of nodes in the graph: " << graph_viewer.NumberOfNodes()
                        << " number of nodes supported by DNNL: " << num_of_supported_nodes
                        << std::fixed << std::setprecision(2) << " (" << percent_dnnl << "%)";
  }

  if (dump_subgraphs_) {
    auto model = graph_viewer.CreateModel(*GetLogger());
    auto model_proto = model->ToProto();
    graph_viewer.ToProto(*model_proto->mutable_graph(), false, true);
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    HashValue model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    std::fstream dump("DNNL_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id) + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto->SerializeToOstream(dump);
  }

  return result;
}

Status DNNLExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                      std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Build tranformer
  auto transformer = std::make_unique<ort_dnnl::New_DnnlGraphTransformer>();
  // Iterate over each supported subgraph
  for (auto& fused_node_graph : fused_nodes_and_graphs) {
    // Get the view to intenal structure of the ONNX graph
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    // Get info of the whole graph
    const Node& fused_node = fused_node_graph.fused_node;

    if (dump_subgraphs_) {
      auto model = graph_body_viewer.CreateModel(*GetLogger());
      auto model_proto = model->ToProto();
      graph_body_viewer.ToProto(*model_proto->mutable_graph(), false, true);
      model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
      std::fstream dump(fused_node.Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto->SerializeToOstream(dump);
    }

    //auto start = std::chrono::high_resolution_clock::now();
    // Build a DnnlGraph from the ONNX graph view
    auto dnnl_graph = std::make_unique<ort_dnnl::New_DnnlGraph>(fused_node.Name(), graph_body_viewer);
    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //build_time += duration.count();
    //printf("------------ New graph build time: %lld ms ------------\n", build_time);
    //transformer->Apply(*dnnl_graph, graph_body_viewer);

    // Store the names of the inputs and outputs so we can get them later
    // we need to do this now because at this point the ONNX graph does not have
    // the data handles with the info for the graph, so we have to store the names so 
    // later we can use then to get the data handles and set them to out graph
    const auto& input_defs = fused_node.InputDefs();
    std::vector<std::string> onnx_input_names(input_defs.size());
    for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
      onnx_input_names[i] = input_defs[i]->Name();
    }
    const auto& output_defs = fused_node.OutputDefs();
    std::vector<std::string> onnx_output_names(output_defs.size());
    for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
      onnx_output_names[i] = output_defs[i]->Name();
    }
    
    // Store current graph info
    graphs_[fused_node.Name()] = {std::move(dnnl_graph), std::move(onnx_input_names), std::move(onnx_output_names)};

    // Build computing functions for the graph execution
    NodeComputeInfo compute_info;

    // Used to get a graph by means of the FunctionState* ptr
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = graphs_[context->node_name].graph.get();
      return 0;
    };
    // Unused
    compute_info.release_state_func = [](FunctionState state) {
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [this](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      Ort::KernelContext ctx(context);

      // Cast to get ptr to the graph built previously
      ort_dnnl::New_DnnlGraph* graph = reinterpret_cast<ort_dnnl::New_DnnlGraph*>(state);
      auto graph_io = &graphs_.at(graph->Name());
      // Get the inputs for the graph 
      const size_t subgraph_num_inputs = graph_io->input_names.size();
      const size_t subgraph_num_outputs = graph_io->output_names.size();
      const size_t context_num_outputs = ctx.GetOutputCount();
      const size_t context_num_inputs = ctx.GetInputCount();

      std::unordered_map<std::string, ort_dnnl::New_OnnxTensorData> inputs;
      inputs.reserve(subgraph_num_inputs);
      for (size_t i = 0; i < context_num_inputs; i++) {
        auto input_name = graph_io->input_names.at(i);  
        auto input_tensor = ctx.GetInput(i);
        auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        // dnnl expectes non-const data
        void* inputBuffer = const_cast<void*>(input_tensor.GetTensorRawData());
        inputs.emplace(input_name,    
                      ort_dnnl::New_OnnxTensorData{
                          ort_dnnl::New_OnnxTensorInfo{tensor_info.GetElementType(), shape},
                          inputBuffer,
                      });
      }

      //lock each subgraph_primitive as multiple threads have shared memories
      {
        // Lock the graph
        std::unique_lock<OrtMutex> lock(graph->GetMutex());
        // Compile
        //printf("------------ Compiling graph: %s ------------ \n", graph->Name().c_str());
        //auto start = std::chrono::high_resolution_clock::now();
        graph->Compile(inputs);
        //auto stop = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //comp_time += duration.count();
        //printf("------------ New graph compilation time: %lld ms ------------ \n", comp_time);
        // Get output info
        std::unordered_map<std::string, ort_dnnl::New_OnnxTensorData> outputs;
        outputs.reserve(subgraph_num_outputs);
        for (size_t i = 0; i < context_num_outputs; i++) {
          auto& output_name = graphs_[graph->Name()].output_names.at(i);
          auto output_md = graph->GetTensor(output_name)->MemoryDesc();  // subgraph_primitive->GetOutputInfo(output_name);
          auto output_shape = output_md.dims();
          //if an output is a scalar, onednn internally uses tensor representation (eg, (1,1,...))
          //but allocating an output with no shape instead of the equivalent tensorshape to avoid shape mismatch
          if (graph->GetTensor(output_name)->IsScalar()) {
            output_shape.clear();
          }
          auto output_tensor =
              ctx.GetOutput(i, output_shape.data(), output_shape.size());
          auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
          auto shape = tensor_info.GetShape();
          void* output_buffer = output_tensor.GetTensorMutableRawData();
          outputs.emplace(output_name,
                          ort_dnnl::New_OnnxTensorData{
                              ort_dnnl::New_OnnxTensorInfo{tensor_info.GetElementType(), shape},
                              output_buffer,
                          });
        }
        // Set data handlers
        graph->SetDataHandles(inputs, outputs);
        //start = std::chrono::high_resolution_clock::now();
        auto x = graph->Predict();
        //stop = std::chrono::high_resolution_clock::now();
        //duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //pred_time += duration.count();
        //printf("------------ New graph prediction time: %lld ms ------------ \n\n", pred_time);
        return x;  // subgraph_primitive->Predict(inputs, outputs);
      }
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }
  return Status::OK();
}

}  // namespace onnxruntime
