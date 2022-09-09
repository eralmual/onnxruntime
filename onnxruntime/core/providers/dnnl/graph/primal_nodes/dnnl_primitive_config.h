// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"

using DnnlInputs = std::unordered_map<int, dnnl::memory>;
using DnnlPrimitive = std::pair<dnnl::primitive, DnnlInputs>;

namespace onnxruntime {
namespace ort_dnnl {

// Auxiliar struct to aid post op generation
struct DnnlPrimitiveConfig {
  // Stores the name of the source primitive, used to determine compatibility
  std::string src_primitive = "";
  // When doing post-ops, this is useful to determine if the shapes are compatible
  dnnl::memory::dims src_out_shape = {};
  // If we detected a type cast then have the src primitive inout and output type to use implicit casting
  dnnl::memory::data_type src_in_type = dnnl::memory::data_type::undef;
  dnnl::memory::data_type src_out_type = dnnl::memory::data_type::undef;
  // Store the post-op list
  dnnl::post_ops post_ops;
  // Store the primitve attributes
  dnnl::primitive_attr prim_attr;
  // Number of postops already added
  size_t num_post_ops = 0;
  // Index of the input that is beign accumulated by the post op
  int input_idx = -1;
  // Store the input map for extended post ops
  DnnlInputs input_map = {};

  // Define small constructor for easier instantiation
  DnnlPrimitiveConfig() = default;
};


}  // namespace ort_dnnl
}  // namespace onnxruntime