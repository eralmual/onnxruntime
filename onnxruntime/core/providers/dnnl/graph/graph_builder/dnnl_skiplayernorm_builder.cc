#include "dnnl_graph_builder.h"


namespace onnxruntime::ort_dnnl {

enum InputTensors_SkipLayerNorm : int {
  IN_INPUT = 0,
  IN_SKIP = 1,
  IN_GAMMA = 2,
  IN_BETA = 3,      // Optional
  IN_BIAS = 4       // Optional
};

enum OutputTensors_SkipLayerNorm : int {
  OUT_OUTPUT = 0,
  OUT_MEAN = 1,        // Optional
  OUT_INV_STD_VAR = 2  // Optional
};

void DnnlGraphBuilder::BuildSkipLayerNorm(const Node* node,
                                            DnnlTensorPtrVector& inputs,
                                            DnnlTensorPtrVector& outputs){

  // Generate the container
  auto onnx_op = UniqueVectorOfUniqueNodes();

  // Get names
  auto node_name = node->Name();
  auto skip_name = node_name + "_Decomposed_0_Add";
  auto bias_name = node_name + "_Decomposed_1_Bias";
  auto ln_name = node_name + "_Decomposed_2_LayerNorm";
  std::string x_tensor_name;

  // Check if beta and exists
  auto beta_exists = false;
  if (inputs->size() >= 4) {
    beta_exists = (*(inputs->at(IN_BETA)))->Exists();
  }
  // Check if bias and exists
  auto bias_exists = false;
  if (inputs->size() >= 5) {
    bias_exists = (*(inputs->at(IN_BIAS)))->Exists();
  }

  {
    // Create skip I/O
    auto skip_out_name = skip_name + "_out";
    auto skip_inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    auto skip_outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    // Get Input tensors 
    auto src_0_tensor = *(inputs->at(IN_INPUT));
    auto src_1_tensor = *(inputs->at(IN_SKIP));
    // Create the Add output tensor and store it
    tensors_.insert({skip_out_name, dnnl_make_tensor(*src_0_tensor, skip_out_name, false)});
    // Add the I/O in the expected order
    skip_inputs->push_back(tensors_.at(src_0_tensor->Name()).get());  // IN_INPUT
    skip_inputs->push_back(tensors_.at(src_1_tensor->Name()).get());  // IN_SKIP
    skip_outputs->push_back(tensors_.at(skip_out_name).get());        // IN_INTERMEDIATE_SKIPED
    // Create the Add node
    nodes_.emplace_back(std::make_unique<DnnlBinaryNode>("Add", skip_name, skip_inputs, skip_outputs));
    // Update x tensor name
    x_tensor_name = skip_out_name;
  }

  {
    if (bias_exists) {
      // Create the bias I/O
      auto bias_out_name = bias_name + "_out";
      auto bias_inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
      auto bias_outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
      // Get Input tensors
      auto bias_tensor = *(inputs->at(IN_BIAS));
      auto src_0_tensor = *(tensors_.at(x_tensor_name));
      // Create the Add output tensor and store it
      tensors_.insert({bias_out_name, dnnl_make_tensor(*src_0_tensor, bias_out_name, false)});
      // Add the I/O in the expected order
      bias_inputs->push_back(tensors_.at(src_0_tensor->Name()).get());  // IN_INTERMEDIATE_SKIPED
      bias_inputs->push_back(tensors_.at(bias_tensor->Name()).get());   // IN_BIAS
      bias_outputs->push_back(tensors_.at(bias_out_name).get());        // IN_INTERMEDIATE_SKIPED_BIASED
      // Create the Add node
      nodes_.emplace_back(std::make_unique<DnnlBinaryNode>("Add", bias_name, bias_inputs, bias_outputs));
      // Update x tensor name 
      x_tensor_name = bias_out_name;
    }
  }

  {
    // Create Layer Normalization I/O
    auto ln_inputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    auto ln_outputs = std::make_shared<std::vector<DnnlTensorPtr*>>();
    // Get Input tensors
    auto x_tensor = *(tensors_.at(x_tensor_name));
    auto gamma_tensor = *(inputs->at(IN_GAMMA));
    // Add the I/O in the expected order
    ln_inputs->push_back(tensors_.at(x_tensor->Name()).get());        // IN_INTERMEDIATE_SKIPED_BIASED
    ln_inputs->push_back(tensors_.at(gamma_tensor->Name()).get());    // IN_GAMMA
    if (beta_exists) {
      auto beta_tensor = *(inputs->at(IN_BETA));
      ln_inputs->push_back(tensors_.at(beta_tensor->Name()).get());   // IN_BETA
    }
    // In case we have multiple outputs
    for (int i = 0; i < outputs->size(); ++i) {
      // Get the tensor
      auto out_tensor = *(outputs->at(i));
      // Check if the tensor exists 
      if (out_tensor->Exists()) {
        // Add it to the list
        ln_outputs->push_back(tensors_.at(out_tensor->Name()).get());
      } else{
        // Add empty node
        ln_outputs->push_back(tensors_.at("Empty_tensor").get());
      }
      
    }
    // Create the Layer Norm node
    nodes_.emplace_back(
        std::make_unique<DnnlLayerNormNode>("LayerNormalization", ln_name, ln_inputs, ln_outputs, GetLayerNormEpsilon(node)));
  }
}

float DnnlGraphBuilder::GetLayerNormEpsilon(const Node* node) {
  auto& node_attr = node->GetAttributes();
  auto epsilon_attr = node_attr.find("epsilon");
  if (epsilon_attr != node_attr.end() &&
      epsilon_attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
    return epsilon_attr->second().f();
  } else {
    ORT_THROW("[oneDNN EP] Error: The 'epsilon' attribute for the ", node->Name(), " node was not provided");
  }
}

}  // namespace onnxruntime::ort_dnnl