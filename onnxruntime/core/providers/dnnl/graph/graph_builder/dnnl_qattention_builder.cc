#include "core/providers/dnnl/graph/graph_builder/dnnl_graph_builder.h"


namespace onnxruntime::ort_dnnl {

enum QAttentionInputTensors : int {
  INPUT = 0,
  WEIGHTS = 1,
  BIAS = 2,
  INPUT_SCALE = 3,
  WEIGHTS_SCALE = 4,
  MASK_INDEX = 5,
  INPUT_ZP = 6,
  WEIGHTS_ZP = 7,
  PAST = 8  // not suppoted
};
enum QAttentionOutputTensors : int {
  OUTPUT = 0,
  PRESENT = 1  // not supported
};

void DnnlGraphBuilder::BuildQAttention(const Node* node,
                                       DnnlTensorPtrVector& inputs,
                                       DnnlTensorPtrVector& outputs) {
  ORT_UNUSED_PARAMETER(node);
  ORT_UNUSED_PARAMETER(inputs);
  ORT_UNUSED_PARAMETER(outputs);
  // Start by calculating the total scale
  /*std::shared_ptr<New_DnnlTensor> scale_tensor;
  if (dnnl_node->GetInput(INPUT_SCALE)->Exists() && dnnl_node->GetInput(WEIGHTS_SCALE)->Exists()) {
    // If both scales exists build a Mul node and calculate them
    auto scale_node = std::make_unique<New_DnnlNode>("Mul");
    auto in_tensor = dnnl_node->GetInput(INPUT_SCALE);
    // Add the I/O in the expected order
    scale_node->SetInput(in_tensor);
    scale_node->SetInput(dnnl_node->GetInput(WEIGHTS_SCALE));
    // Create the output tensor
    scale_tensor = std::make_shared<New_DnnlTensor>(*in_tensor, "QAtt_Scale_Mul_out", false);
    // Set as output
    scale_node->SetOutput(scale_tensor);
    // Add the op to the graph
    nodes.emplace_back(std::move(scale_node));


  } else if (dnnl_node->GetInput(INPUT_SCALE)->Exists()) {
    // If only INPUT_SCALE scale exists then use it
    scale_tensor = dnnl_node->GetInput(INPUT_SCALE);
  } else if (dnnl_node->GetInput(WEIGHTS_SCALE)->Exists()) {
    // If only WEIGHTS_SCALE scale exists then use it
    scale_tensor = dnnl_node->GetInput(WEIGHTS_SCALE);
  }

  // Create the MatMulInteger node
  auto matmulint_node = std::make_unique<New_DnnlNode>("MatMulInteger");
  // TODO ADD BF16 DETECTION
  // Create the MatMulInteger output tensor, we dont care about the dims since primitive creation calcs their own
  auto qkv_tensor = std::make_shared<New_DnnlTensor>(dnnl::memory::dims(), dnnl::memory::data_type::f32,
                                                    "QAtt_QKV", false);
  // Add the I/O in the expected order
  matmulint_node->SetInput(dnnl_node->GetInput(INPUT));       // IN_A
  matmulint_node->SetInput(dnnl_node->GetInput(WEIGHTS));     // IN_B
  matmulint_node->SetInput(dnnl_node->GetInput(INPUT_ZP));    // IN_A_ZERO_POINT
  matmulint_node->SetInput(dnnl_node->GetInput(WEIGHTS_ZP));  // IN_B_ZERO_POINT
  matmulint_node->SetInput(scale_tensor);                     // IN_OUTPUT_SCALE
  matmulint_node->SetOutput(qkv_tensor);                      // OUT_Y
  // Add MatMulInteger to the graph
  nodes.emplace_back(std::move(matmulint_node));

  // Create the Add node for the bias
  auto add_bias_node = std::make_unique<New_DnnlNode>("Add");
  // Create the Add bias output tensor
  auto qkv_biased_tensor = std::make_shared<New_DnnlTensor>(*qkv_tensor, "QAtt_QKV_Add_Bias", false);
  // Add the I/O in order, add bias last since we expect it to be the one that needs to be broadcasted
  add_bias_node->SetInput(qkv_tensor);                  // IN_A
  add_bias_node->SetInput(dnnl_node->GetInput(BIAS));   // IN_B
  add_bias_node->SetOutput(qkv_biased_tensor);          // OUT_Y
  // Add the Add to the graph
  nodes.emplace_back(std::move(add_bias_node));

  // If it exists get the mask ready
  if (dnnl_node->GetInput(MASK_INDEX)->Exists()) {
    // Create the Add node for the bias
    auto linear_mask_node = std::make_unique<New_DnnlNode>("Linear");
    // Create the linear transformed mask output tensor
    auto linear_mask_tensor = std::make_shared<New_DnnlTensor>(*dnnl_node->GetInput(MASK_INDEX),
                                                              "QAtt_QKV_Mask_linear", false);
    // Set I/O
    linear_mask_node->SetInput(dnnl_node->GetInput(MASK_INDEX));  // IN_DATA
    linear_mask_node->SetOutput(linear_mask_tensor);              // OUT_Y
    // Add attributes
    linear_mask_node->SetAttribute("alpha", "10000.0");
    linear_mask_node->SetAttribute("beta", "-10000.0");
    // Add the Add to the graph
    nodes.emplace_back(std::move(linear_mask_node));

    // Create the Cast node to F32
    auto lcast_mask_node = std::make_unique<New_DnnlNode>("Cast");
    // Create the casted mask output tensor
    auto lcast_mask_tensor = std::make_shared<New_DnnlTensor>(linear_mask_tensor->Dims(), dnnl::memory::data_type::f32,
                                                               "QAtt_QKV_Mask_LCasted", false);
    // Set I/O
    lcast_mask_node->SetInput(linear_mask_tensor);    // IN_INPUT
    lcast_mask_node->SetOutput(lcast_mask_tensor);    // OUT_OUTPUT
    // Add attributes
    lcast_mask_node->SetAttribute("to", "1");
  }*/
}


}  // namespace onnxruntime::ort_dnnl