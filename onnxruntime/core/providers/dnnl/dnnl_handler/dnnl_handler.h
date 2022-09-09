// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlHandler {
 public:

  DnnlHandler();

  const dnnl::engine& GetEngine();
  const dnnl::engine& GetCPUEngine();
  dnnl::stream GetStream();
  dnnl::memory::format_tag GetDnnlFormat(size_t dim_size);
 
 private:
  dnnl::engine cpu_engine_;
  dnnl::engine gpu_engine_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime