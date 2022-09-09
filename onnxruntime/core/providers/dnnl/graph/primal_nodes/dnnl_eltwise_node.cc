// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_eltwise_node.h"

namespace onnxruntime {
namespace ort_dnnl {

std::unique_ptr<std::vector<DnnlPrimitive>> DnnlEltwiseNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                                         const dnnl::engine& dnnl_engine) {
  // Get the input memory
  auto src_tensor = GetInput(IN_X);
  // Get output tensor
  auto dst_tensor = GetOutput(OUT_Y);

  // Add post ops
  dnnl::primitive_attr prim_attr;
  prim_attr.set_post_ops(post_op_config.post_ops);

  // Create the primitive descriptor
  auto eltwise_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                  algorithm_, src_tensor->MemoryDesc(), GetAlpha(), GetBeta());
  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_desc, prim_attr, dnnl_engine);
  // Generate the memory
  dst_tensor->SetMemory(std::make_unique<dnnl::memory>(eltwise_pd.dst_desc(), dnnl_engine));
  // Create primititve
  auto eltwise_prim = dnnl::eltwise_forward(eltwise_pd);
  // Create the inputs
  post_op_config.input_map[DNNL_ARG_SRC] = src_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_DST] = dst_tensor->Memory();

  // This is to comply with ONNX scalar handling, if sources are scalar then so is output
  if (src_tensor->IsScalar()) {
    dst_tensor->SetAsScalar();
  }

  // Create the primitive sequence object and append primitives
  auto primitive_seq = std::make_unique<std::vector<DnnlPrimitive>>();
  primitive_seq->emplace_back(std::make_pair(eltwise_prim, post_op_config.input_map));

  // Create the Primitive bundle object
  return primitive_seq;
}

void DnnlEltwiseNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) {
  // Append the primitive
  post_op_config.post_ops.append_eltwise(1.0f, algorithm_, GetAlpha(),GetBeta());
  // Increase the post-op counter
  ++post_op_config.num_post_ops;
}

bool DnnlEltwiseNode::IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) {

   // The src_primitive MUST support this primitive as a post-op
  if (!primitive_utils::IsPostOpCompatible(post_op_config->src_primitive, primitive_name_)) {
    return false;
  }
  // BatchNormalization is only compatible with ReLU
  else if ((post_op_config->src_primitive == "BatchNormalization") &&
           (OpType() != "Relu")) {
    return false;
  }

  return true;
}

void DnnlEltwiseNode::BuildInPlace(DnnlPrimitiveConfig* post_op_config) {
  // If we need to cast then no
  // We can add casting support later
  if (post_op_config->src_out_type != dnnl::memory::data_type::undef) {
    return;
  }
}

void DnnlEltwiseNode::CalculateOutputDims() {
  // Set output dims eq to src dims
  output_dims_ = std::make_unique<dnnl::memory::dims>(GetInput(IN_X)->Dims());
}

float DnnlEltwiseNode::GetAlpha() {
  // For the pow operator, we need to hard code alpha to 1
  if (algorithm_ == dnnl ::algorithm::eltwise_pow) {
    return 1.0f;
  }

  // Defaut alpha value for oneDNN primitives
  float alpha = 0.0;

  // Check if the attr have an alpha value
  auto is_alpha_found = false;
  // Run this check only if we have the attributes
  auto attributes = GetAttributes();
  if (attributes) {
    auto attr = attributes->find("alpha");
    if (attr != attributes->end()) {
      // When we create artificial attributes, we manually have to specify them as strings so take that into account
      if (attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
        alpha = std::stof(attr->second().s());
      } else {
        alpha = attr->second().f();
      }
      // Check as found
      is_alpha_found = true;
    }
  }

  // Check for ONNX defined default alpha value for other algorithms
  if (algorithm_ == dnnl::algorithm::eltwise_elu) {
    // If no alpha is provided set the ONNX defined default value
    return is_alpha_found ? alpha : 1.0f;

  } else if (algorithm_ == dnnl ::algorithm::eltwise_relu && (OpType() == "LeakyRelu")) {
    // Need to check operator since both Relu and LeakyRelu are covered by algorithm::eltwise_relu
    return is_alpha_found ? alpha : 0.01f;
  }

  return alpha;
}

float DnnlEltwiseNode::GetBeta() {
  // For pow the beta is provided as an input
  if (algorithm_ == dnnl ::algorithm::eltwise_pow) {
    return GetInput(IN_Y)->GetDataAsVector<float>().at(0);
  }
  // Defaut alpha value for oneDNN primitives
  float beta = 0.0;
  // Run this check only if we have the attributes
  auto attributes = GetAttributes();
  if (attributes) {
    auto attr = attributes->find("beta");
    if (attr != attributes->end()) {
      // When we create the attributes, we manually have to specify them as strings so take that into account
      if (attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
        beta = std::stof(attr->second().s());
      } else {
        beta = attr->second().f();
      }
    }
  }
  // Return beta
  return beta;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime