#include "dnnl_graph_builder.h"

namespace onnxruntime::ort_dnnl {

enum InputTensors_BiasGelu : int {
  IN_A = 0,
  IN_B = 1,

};

enum OutputTensors_BiasGelu : int {
  OUT_C = 0,
};

void DnnlGraphBuilder::BuildBiasGelu(const Node* node,
                                          DnnlTensorPtrVector& inputs,
                                          DnnlTensorPtrVector& outputs) {
  // Generate the container
  auto onnx_op = UniqueVectorOfUniqueNodes();

  // Get names
  auto node_name = node->Name();
  auto bias_name = node_name + "_Decomposed_0_Bias";
  auto gelu_name = node_name + "_Decomposed_1_GeLU";
  std::string x_tensor_name;

  {
    // Create the bias I/O
    auto bias_out_name = bias_name + "_out";
    auto bias_inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    auto bias_outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    // Get Input tensors
    auto src_0_tensor = *(inputs->at(IN_A));
    auto src_1_tensor = *(inputs->at(IN_B));
    // Create the Add output tensor and store it
    tensors_.insert({bias_out_name, dnnl_make_tensor(*src_0_tensor, bias_out_name, false)});
    // Add the I/O in the expected order
    bias_inputs->push_back(tensors_.at(src_0_tensor->Name()).get());  // IN_A
    bias_inputs->push_back(tensors_.at(src_1_tensor->Name()).get());  // IN_B
    bias_outputs->push_back(tensors_.at(bias_out_name).get());        // IN_INTERMEDIATE_BIASED
    // Create the Add node
    nodes_.emplace_back(std::make_unique<DnnlBinaryNode>("Add", bias_name, bias_inputs, bias_outputs));
    // Update x tensor name
    x_tensor_name = bias_out_name;
  }

  {
    // Create Layer Normalization I/O
    auto gelu_inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    auto gelu_outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    // Get I/O tensors
    auto x_tensor = *(tensors_.at(x_tensor_name));
    auto out_tensor = *(outputs->at(OUT_C));
    // Add the I/O in the expected order
    gelu_inputs->push_back(tensors_.at(x_tensor->Name()).get());      // IN_INTERMEDIATE_BIASED
    gelu_outputs->push_back(tensors_.at(out_tensor->Name()).get());   // OUT_C
    // Create the Bias GeLU node
    nodes_.emplace_back(std::make_unique<DnnlEltwiseNode>("Gelu", gelu_name, gelu_inputs, gelu_outputs));
  }
}

}  // namespace onnxruntime::ort_dnnl