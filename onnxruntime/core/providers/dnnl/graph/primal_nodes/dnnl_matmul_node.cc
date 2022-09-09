// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_matmul_node.h"

#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

namespace onnxruntime {
namespace ort_dnnl {

std::unique_ptr<std::vector<DnnlPrimitive>> DnnlMatMulNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                                        const dnnl::engine& dnnl_engine) {
  // Create the primitive sequence object
  auto primitive_seq = std::make_unique<std::vector<DnnlPrimitive>>();

  // Get the I/O tensors
  auto src_tensor = GetInput(IN_A);
  auto weights_tensor = GetInput(IN_B);
  auto dst_tensor = GetOutput(OUT_Y);
  // Padd dimensions
  auto padded_srcs_md = primitive_utils::PaddSourcesEqualy( src_tensor->GenerateInputMemoryDesc(), 
                                                            weights_tensor->GenerateInputMemoryDesc());

  // Generate dst md 
  auto dst_md = dnnl::memory::desc(*output_dims_, 
      post_op_config.src_out_type == dnnl::memory::data_type::undef ? // If undef then output dt else config value
                  dst_tensor->DataType() : post_op_config.src_out_type, dnnl::memory::format_tag::any);


  // Define attributes here as they are needed later
  dnnl::primitive_attr prim_attr;
  // TODO ADD SUPPORT FOR SCALE OUTPUT, NEEDED FOR QATTENTION

  // Add zero points in case we are running MatMulInteger
  if (OpType() == "MatMulInteger") {
    // Start evaluating zero points
    CastZeroPoint(*GetInput(IN_A_ZERO_POINT), primitive_seq.get());
    CastZeroPoint(*GetInput(IN_B_ZERO_POINT), primitive_seq.get());
    AddZeroPoint(IN_A_ZERO_POINT, prim_attr, post_op_config);
    AddZeroPoint(IN_B_ZERO_POINT, prim_attr, post_op_config);
  }

  // Add post ops
  prim_attr.set_post_ops(post_op_config.post_ops);
  // Generate MatMul descriptor and add post ops
  auto matmul_d = dnnl::matmul::desc(padded_srcs_md.first, padded_srcs_md.second, dst_md);
  auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, prim_attr, dnnl_engine);
  // Reorder inputs to set the best performing layout if necesary
  src_tensor->ReorderMemory(matmul_pd.src_desc(), primitive_seq.get(), &dnnl_engine);
  weights_tensor->ReorderMemory(matmul_pd.weights_desc(), primitive_seq.get(), &dnnl_engine);
  // Set the output memory to the output tensor
  dst_tensor->SetMemory(std::make_unique<dnnl::memory>(matmul_pd.dst_desc(), dnnl_engine));
  // Create primitive
  auto matmul_prim = dnnl::matmul(matmul_pd);
  // Create the inputs
  post_op_config.input_map[DNNL_ARG_SRC] =      src_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_WEIGHTS] =  weights_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_DST] =      dst_tensor->Memory();

  /*printf("----------------------- Tensors at primitive time -----------------------\n");
  //printf("src tensor ptr: %p\n", src_tensor.get());
  //printf("weights tensor ptr: %p\n", weights_tensor.get());
  for (auto tensor : {src_tensor, weights_tensor}) {
    if (tensor->Name() != "") {
      auto mem = static_cast<int8_t*>(tensor->Memory().get_data_handle());
      auto limit = tensor->Dims()[0] > 2 ? 10 : 1;
      printf("%s = {", tensor->Name().c_str());
      for (int i = 0; i < limit; ++i) {
        printf("%i, ", static_cast<int>(mem[i]));
      }
      printf("}\n");
    }
  }
  printf("dst tensor ptr: %p\n", dst_tensor.get());
  printf("-----------------------------------------------------------------------------\n");*/

  // Append primitives
  primitive_seq->emplace_back(std::make_pair(matmul_prim, post_op_config.input_map));

  // Create the Primitive bundle object
  return primitive_seq;
}

void DnnlMatMulNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
  // Matmul is post op to no one
  ORT_THROW("[oneDNN EP] Error: Tried to generate a MatMul post-op");
}

bool DnnlMatMulNode::IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
  return false;
}

void DnnlMatMulNode::BuildInPlace(DnnlPrimitiveConfig* post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
}

void DnnlMatMulNode::CalculateOutputDims() {
  // Padd dimensions to calc output to evaluate primitive compatibility. Consider saving this value?
  auto src_dims = GetInput(IN_A)->Dims();
  auto weights_dims = GetInput(IN_B)->Dims();
  primitive_utils::PaddSourcesEqualy(src_dims, weights_dims);

  // Generate output dims
  output_dims_ = std::make_unique<dnnl::memory::dims>(src_dims);
  output_dims_->pop_back();
  output_dims_->emplace_back(weights_dims.back());
  for (size_t i = 0; i < output_dims_->size() - 2; i++) {
    if (output_dims_->at(i) == 1) {
      output_dims_->at(i) = weights_dims.at(i);
    }
  }
}

inline void DnnlMatMulNode::CastZeroPoint( New_DnnlTensor& zero_point,
                                            std::vector<DnnlPrimitive>* prim_list) {
  // We only cast when needed, for constant initializers, this is not necesary
  if (zero_point.Exists() 
      && (zero_point.DataType() != dnnl::memory::data_type::s32)
      && !zero_point.IsConstantInitializer() /*&& !zero_point.IsInitialized()*/) {
    // Store dims since we'll be using them
    auto zp_dims = zero_point.Dims();
    // Generate target descriptor and mem
    auto dst_md = dnnl::memory::desc(zp_dims, dnnl::memory::data_type::s32, FormatMap.at(zp_dims.size()));
    zero_point.ReorderMemory(dst_md, prim_list);
  }
}

inline void DnnlMatMulNode::AddZeroPoint(int tensor_idx,
                                          dnnl::primitive_attr& prim_attr,
                                          DnnlPrimitiveConfig& post_op_config) {
  // Get the right primitive index 
  int primitive_idx;
  if (tensor_idx == IN_A_ZERO_POINT) {
    primitive_idx = DNNL_ARG_SRC;
  } else if (tensor_idx == IN_B_ZERO_POINT) {
    primitive_idx = DNNL_ARG_WEIGHTS;
  } else {
    ORT_THROW("[oneDNN] Error: Trying to create a MatMulInteger zero point with invalid tensor index");
  }

  // Get zero point related inputs auto a_zp = GetInput(IN_A_ZERO_POINT);
  auto zero_point_tensor = GetInput(tensor_idx);

  // Check if the zero point exists
  if (zero_point_tensor->Exists()) {
    // If we have constant initializers with dim == 1, we could add support for more in the future
    if (zero_point_tensor->IsConstantInitializer()
        && (zero_point_tensor->Dims().size() == 1)
        && (zero_point_tensor->Dims()[0] == 1)) {
      // Add the src zero pont as a runtime parameter
      auto zp_vector = zero_point_tensor->GetDataAsVector<int32_t>();
      // If our zp is just one value, mask is 0, else is 2
      //auto mask = zp_vector.size() > 1 ? 2 : 0;
      prim_attr.set_zero_points(primitive_idx, /* mask */ 0, zp_vector);
    } else {
      // Add the src zero point as a runtime parameter
      prim_attr.set_zero_points(primitive_idx, /* mask */ 0, {DNNL_RUNTIME_S32_VAL});
      // Add the zp to the input memory map
      post_op_config.input_map[DNNL_ARG_ATTR_ZERO_POINTS | primitive_idx] = zero_point_tensor->Memory();
    }
  }
}

}  // namespace ort_dnnl
}  // namespace onnxruntime