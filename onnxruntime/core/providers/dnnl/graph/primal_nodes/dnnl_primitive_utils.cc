// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_primitive_utils.h"

namespace onnxruntime::ort_dnnl::primitive_utils {

// Mapping functions

std::string OrtOperatorToDnnlPrimitiveName(const std::string& op) {
  return onnx_op_to_onednn_primitive.at(op);
}

dnnl::algorithm OrtOperatorToDnnlAlgorithm(const std::string& op) {
  return onnx_op_to_onednn_algorithm.at(op);
}

bool IsSimpleOperator(const std::string& op) {
  return onnx_op_to_onednn_primitive.count(op);
}

bool IsPostOpCompatible(const std::string& source_op, const std::string& next_op) {
  // Get src compatible ops
  auto src_compat_ops = onednn_postop_rules.at(source_op);
  // Check if the next_op type of primitive is on the compatible list
  return src_compat_ops.count(next_op);
}

bool IsImplicitCastCompatible(const std::string& primitive, const dnnl_dt& src_dt, const dnnl_dt& dst_dt) {
  // Start by checking if there is a cast 
  if (src_dt == dst_dt) {
    return true;
  }  
  // Get the primitive compatibility map
  auto prim_compat_map = onednn_src_dst_dt_rules.at(primitive);
  // Check if the key is in the map
  if (prim_compat_map.count(src_dt)) {
    // If so then the result is whether or not the dst is in the supported dt set
    return prim_compat_map.at(src_dt).count(dst_dt);

  // If the src is not on the map then check for empty map wich means universal dt support
  } else {
    return prim_compat_map.empty();
  }
}


// Padding functions
std::pair<dnnl::memory::desc, dnnl::memory::desc> PaddSourcesEqualy(const dnnl::memory::desc& src_0_md,
                                                                    const dnnl::memory::desc& src_1_md) {
  // Get srcs dims
  auto src_0_dims = src_0_md.dims();
  auto src_1_dims = src_1_md.dims();
  // Padd dimensions if necesary
  PaddSourcesEqualy(src_0_dims, src_1_dims);
  // Return padded mds
  return std::make_pair(src_0_md.reshape(src_0_dims), src_1_md.reshape(src_1_dims));
}

void PaddSourcesEqualy(dnnl::memory::dims& src_0_dims, dnnl::memory::dims& src_1_dims) {
  // Padd dimensions if necesary
  if (src_0_dims.size() != src_1_dims.size()) {
    // If src_1 is smaller add padding
    while (src_0_dims.size() < src_1_dims.size()) {
      src_0_dims.insert(src_0_dims.begin(), 1);
    }
    // If src_2 is smaller add padding
    while (src_0_dims.size() > src_1_dims.size()) {
      src_1_dims.insert(src_1_dims.begin(), 1);
    }
  }
}

dnnl::memory::desc Padd(dnnl::memory::desc& target_md, size_t front_pad, size_t back_pad) {
  // Get dims from target
  auto target_dims = target_md.dims();
  // Add front padding
  while (target_dims.size() < front_pad) {
    target_dims.insert(target_dims.begin(), 1);
  }
  // Add back padd
  while (target_dims.size() < back_pad) {
    target_dims.insert(target_dims.end(), 1);
  }
  // Return padded md
  return target_md.reshape(target_dims);
}

bool IsMemoryInExpectedOrtFormat(const dnnl::memory::desc& desc) {
  if (desc.data.format_kind != dnnl_blocked) {
    return false;
  }
  if (desc.data.format_desc.blocking.inner_nblks != 0) {
    return false;
  }
  auto strides = desc.data.format_desc.blocking.strides;
  // if a data format is dnnl_format::abcd... the stride will go from largest to smallest
  // if for example we have a shape {2,3,4} we expect a stride of {12, 4, 1} if it were
  // of dnnl_format::abc if instead the stride were {12, 1, 4} that would be dnnl_format::acb
  // which does not match what is expected from Onnxruntime.
  for (size_t i = 1; i < desc.dims().size(); ++i) {
    if (strides[i - 1] < strides[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace onnxruntime::ort_dnnl::primitive_utils
