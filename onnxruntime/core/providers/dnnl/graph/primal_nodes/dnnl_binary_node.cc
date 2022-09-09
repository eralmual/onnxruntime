// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_binary_node.h"

#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"


namespace onnxruntime {
namespace ort_dnnl {

std::unique_ptr<std::vector<DnnlPrimitive>> DnnlBinaryNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                                        const dnnl::engine& dnnl_engine) {
  // Create the primitive sequence object
  auto primitive_seq = std::make_unique<std::vector<DnnlPrimitive>>();
  // Get the algorithm 
  auto algorithm = primitive_utils::OrtOperatorToDnnlAlgorithm(OpType());
  // Get the I/O tensors
  auto src_tensor_0 = GetInput(IN_A);
  auto src_tensor_1 = GetInput(IN_B);
  auto dst_tensor =   GetOutput(OUT_Y);
  // Make sure src memory formats are plain
  src_tensor_0->ReorderMemory(src_tensor_0->GeneratePlainMemoryDesc(), primitive_seq.get());
  src_tensor_1->ReorderMemory(src_tensor_1->GeneratePlainMemoryDesc(), primitive_seq.get());
  
  // Padd dimensions
  auto padded_src_md = primitive_utils::PaddSourcesEqualy(src_tensor_0->MemoryDesc(),
                                                          src_tensor_1->MemoryDesc());
  // Place holder
  dnnl::memory::desc dst_md;

  // Check if we can  do the op inplace
  if (is_inplace_) {
    dst_md = dst_tensor->MemoryDesc();
  } else {
    // Generate dst md 
    dst_md = dnnl::memory::desc(*output_dims_, 
        post_op_config.src_out_type == dnnl::memory::data_type::undef ?   // If undef then output dt else config value
          dst_tensor->DataType() : post_op_config.src_out_type, dnnl::memory::format_tag::any);
  }

  // Add post ops
  dnnl::primitive_attr prim_attr;
  prim_attr.set_post_ops(post_op_config.post_ops);
  
  // Create the primitive descriptor
  auto binary_d = dnnl::binary::desc(algorithm, padded_src_md.first, padded_src_md.second, dst_md);
  auto binary_pd = dnnl::binary::primitive_desc(binary_d, prim_attr, dnnl_engine);

  // We generate the memory here because non-inplace ops will force the binary_pd to create a md with the ideal layout
  if (!is_inplace_) {
    // Set the output memory to the output tensor
    dst_tensor->SetMemory(std::make_unique<dnnl::memory>(binary_pd.dst_desc(), dnnl_engine));
  }
  // Create primititve
  auto binary_prim = dnnl::binary(binary_pd);
  // Create the inputs
  post_op_config.input_map[DNNL_ARG_SRC_0] = src_tensor_0->Memory();
  post_op_config.input_map[DNNL_ARG_SRC_1] = src_tensor_1->Memory();
  post_op_config.input_map[DNNL_ARG_DST] =   dst_tensor->Memory();

  // This is to comply with ONNX scalar handling, if sources are scalar then so is output
  if (src_tensor_0->IsScalar() && src_tensor_1->IsScalar()) {
    dst_tensor->SetAsScalar();
  }

  // Append binary prim
  primitive_seq->emplace_back(std::make_pair(binary_prim, post_op_config.input_map));
  
  // Create the Primitive bundle object
  return primitive_seq;
}

void DnnlBinaryNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) {
  
  // Select the input we need
  auto input = !post_op_config.input_idx;
 
  // Get the algorithm
  auto algorithm = primitive_utils::OrtOperatorToDnnlAlgorithm(OpType());  
  // Get src md
  auto src_md = GetInput(input)->MemoryDesc();
  // Padd if having a dim mismatch and src is not scalar
  if (GetInput(input)->Dims().size() < post_op_config.src_out_shape.size()) {
    src_md = primitive_utils::Padd(src_md, post_op_config.src_out_shape.size(), 0);
  }
    
  // Possible improvement: use format any to choose the best layout
  // Add the binary as post-op
  post_op_config.post_ops.append_binary(algorithm, src_md);
  // Add the op input
  post_op_config
    .input_map[DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(post_op_config.num_post_ops)) | DNNL_ARG_SRC_1]
     = GetInput(input)->Memory();
  // Increase the post-op counter
  ++post_op_config.num_post_ops;
}

bool DnnlBinaryNode::IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) {
  // Some required variables
  auto binary_input_idx = !post_op_config->input_idx;
  auto binary_input_dims = GetInput(binary_input_idx)->Dims();

  // The src_primitive MUST support this primitive as a post-op
  if (!primitive_utils::IsPostOpCompatible(post_op_config->src_primitive, primitive_name_)) {
    return false;
  }
  // Dims are compatible as long as the src op has more dims than current operand
  // TODO WE COULD FIX THIS BY ADDING THIS INFO TO THE POST OP CONFIG AND RESIZING BUT WOULD
  // HAVE TO BE CAREFULL SINCE OTHERE OPS COULD BE ADDED BEFORE AND WE WOULD HAVE TO HANDLE IT
  if (!(binary_input_dims.size() <= post_op_config->src_out_shape.size())) {
    return false;
  }
  // oneDNN can only fuse binary post op for the second input. Due to the fact that division and subraction
  // are not associative we can not fuse them if the input for that node is the first input.
  if (((OpType() == "Div") || (OpType() == "Sub")) && 
      (binary_input_idx != IN_A)) {
    return false;
  }
  //if (post_op_config->src_primitive == "MatMul") {
  //  return true;
  //}
  // Only fuse if the broadcasting rule is optimized for IN_B shape
  if (!IsBroadcastPostOpOptimized(binary_input_idx == IN_B ? binary_input_dims : post_op_config->src_out_shape)) {
    return false;
  }

  // If all is ok, then fuse
  return true;
}

void DnnlBinaryNode::BuildInPlace(DnnlPrimitiveConfig* post_op_config) {
  auto src_tensor = GetOutput(IN_A);
  auto dst_tensor = GetOutput(OUT_Y);
  // If we do implicit casting to a type != from inplace target mem the dont support
  if ((post_op_config->src_out_type != dnnl::memory::data_type::undef)
      && (post_op_config->src_out_type != src_tensor->DataType())) {
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
  SetInPlaceOutput(IN_A, OUT_Y);
}

void DnnlBinaryNode::CalculateOutputDims() {
  // Padd dimensions to calc output to evaluate primitive compatibility. Consider saving this value?
  auto src_0_dims = GetInput(IN_A)->Dims();
  auto src_1_dims = GetInput(IN_B)->Dims();
  primitive_utils::PaddSourcesEqualy(src_0_dims, src_1_dims);

  // Generate output dims
  output_dims_ = std::make_unique<dnnl::memory::dims>(src_0_dims);
  for (size_t i = 0; i < output_dims_->size(); i++) {
    if (output_dims_->at(i) == 1) {
      output_dims_->at(i) = src_1_dims[i];
    }
  }
}

bool DnnlBinaryNode::IsBroadcastPostOpOptimized(dnnl::memory::dims& src_dims) {
  // Generate bool flags
  auto is_scalar = true;
  auto is_per_oc = true;
  // Iterate over the tensor 
  for (size_t i = 0; i < src_dims.size(); ++i) {
    // For scalar every dim should be 1
    is_scalar &= (src_dims[i] == 1);
    // For per_oc every dim should be 1 but the second
    if (i == 1) {
      is_per_oc &= (src_dims[i] != 1);
    } else {
      is_per_oc &= (src_dims[i] == 1);
    }
  }

  return is_scalar || is_per_oc;
 }


}  // namespace ort_dnnl
}  // namespace onnxruntime