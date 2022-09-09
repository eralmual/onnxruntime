// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_layernorm_node.h"

#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"


namespace onnxruntime {
namespace ort_dnnl {

DnnlPrimitiveVector DnnlLayerNormNode::GeneratePrimitive( DnnlPrimitiveConfig& post_op_config,
                                                          const dnnl::engine& dnnl_engine) {

  // Get the I/O tensors
  auto src_tensor = GetInput(IN_X);
  auto scale_tensor = GetInput(IN_SCALE);
  auto shift_tensor = GetInput(IN_B);
  auto dst_tensor = GetOutput(OUT_Y);
  // Check if shift exists
  auto shift_exists = shift_tensor->Exists();

  // Source and dst md
  auto src_md = src_tensor->MemoryDesc();

  // X = LayerNorm(X)
  // Check if we are training and need the extra outputs for backprop
  dnnl::prop_kind prop_kind;
#if 0  // defined(ENABLE_TRAINING)
  prop_kind = dnnl::prop_kind::forward_training;
#else
  prop_kind = dnnl::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

  // If beta is available use shift, else only scale
  auto op_flags = dnnl::normalization_flags::use_scale;
  if (shift_exists) {
    op_flags |= dnnl::normalization_flags::use_shift;
  }

  // Get epsilon to avoid zero division
  GetEpsilon();
  // Operation desciptor
  auto lnorm_desc = dnnl::layer_normalization_forward::desc(prop_kind, src_md, epsilon_, op_flags);
  // Primitive desciptor
  auto lnorm_pd = dnnl::layer_normalization_forward::primitive_desc(lnorm_desc, dnnl_engine);
  // Primitive
  auto lnorm_prim = dnnl::layer_normalization_forward(lnorm_pd);
  // We generate the memory here because non-inplace ops will force the binary_pd to create a md with the ideal layout
  if (!is_inplace_) {
    // Set the output memory to the output tensor
    dst_tensor->SetMemory(std::make_unique<dnnl::memory>(lnorm_pd.dst_desc(), dnnl_engine));
  }

  // Define primitive arguments
  post_op_config.input_map[DNNL_ARG_SRC] =    src_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_SCALE] =  scale_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_DST] =    dst_tensor->Memory();

  // Get Beta and add shift if available
  if (shift_exists) {
    post_op_config.input_map[DNNL_ARG_SHIFT] = shift_tensor->Memory();
  }

  // Create the primitive sequence object and append primitives
  auto primitive_seq = std::make_unique<std::vector<DnnlPrimitive>>();
  primitive_seq->emplace_back(std::make_pair(lnorm_prim, post_op_config.input_map));

  // Create the Primitive bundle object
  return primitive_seq;
}

void DnnlLayerNormNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
  // Matmul is post op to no one
  ORT_THROW("[oneDNN EP] Error: Tried to generate a Layer Normalization post-op");
}

bool DnnlLayerNormNode::IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
  return false;
}

void DnnlLayerNormNode::BuildInPlace(DnnlPrimitiveConfig* post_op_config) {
  auto src_tensor = GetOutput(IN_X);
  auto dst_tensor = GetOutput(OUT_Y);
  // If we do implicit casting to a type != from inplace target mem the dont support
  if ((post_op_config->src_out_type != dnnl::memory::data_type::undef) && (post_op_config->src_out_type != src_tensor->DataType())) {
    return;

    // If there is no implicit cast then output dt should be compatible
  } else if (src_tensor->DataType() != dst_tensor->DataType()) {
    return;
  }
  // Data handles are not managed in-place by ORT so dont optimize for graph outputs
  if (dst_tensor->IsGraphOutput()) {
    return;
  }
  // Set primitive as inplace for primitive_build
  is_inplace_ = true;
  // Set input as output
  SetInPlaceOutput(IN_X, OUT_Y);
}

void DnnlLayerNormNode::CalculateOutputDims() {
  // Generate output dims
  output_dims_ = std::make_unique<dnnl::memory::dims>(GetInput(IN_X)->Dims());
}

void DnnlLayerNormNode::GetEpsilon() {
  // Default value according to ONNX spec
  // Run this check only if we have the attributes
  auto attributes = GetAttributes();
  if (attributes) {
    auto attr = attributes->find("epsilon");
    if (attr != attributes->end()) {
      // When we create the attributes, we manually have to specify them as strings so take that into account
      if (attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
        epsilon_ = std::stof(attr->second().s());
      } else {
        epsilon_ = attr->second().f();
      }
    }
  }
  if (epsilon_ == -1) {
    ORT_THROW("[oneDNN EP] Error: The 'epsilon' attribute for the ", Name(), " node was not provided");
  }
}


}  // namespace ort_dnnl
}  // namespace onnxruntime