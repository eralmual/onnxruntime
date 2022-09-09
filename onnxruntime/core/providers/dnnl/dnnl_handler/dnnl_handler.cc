// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_handler.h"

namespace onnxruntime {
namespace ort_dnnl {

DnnlHandler::DnnlHandler() {
  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_cpu)) {
    cpu_engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
  }

  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
    gpu_engine_ = dnnl::engine(dnnl::engine::kind::gpu, 0);
  }
}

const dnnl::engine& DnnlHandler::GetCPUEngine() {
  return cpu_engine_;
}

const dnnl::engine& DnnlHandler::GetEngine() {
  if (gpu_engine_) {
    return gpu_engine_;
  }
  return cpu_engine_;
}

dnnl::stream DnnlHandler::GetStream() {
  return dnnl::stream(GetEngine());
}

dnnl::memory::format_tag DnnlHandler::GetDnnlFormat(size_t dim_size) {
  dnnl::memory::format_tag source_format = dnnl::memory::format_tag::any;
  switch (dim_size) {
    case 1: {
      source_format = dnnl::memory::format_tag::x;
      break;
    }
    case 2: {
      source_format = dnnl::memory::format_tag::nc;
      break;
    }
    case 3: {
      source_format = dnnl::memory::format_tag::ncw;
      break;
    }
    case 4: {
      source_format = dnnl::memory::format_tag::nchw;
      break;
    }
    case 5: {
      source_format = dnnl::memory::format_tag::ncdhw;
      break;
    }
    case 6: {
      source_format = dnnl::memory::format_tag::abcdef;
      break;
    }
    case 7: {
      source_format = dnnl::memory::format_tag::abcdefg;
      break;
    }
    case 8: {
      source_format = dnnl::memory::format_tag::abcdefgh;
      break;
    }
    case 9: {
      source_format = dnnl::memory::format_tag::abcdefghi;
      break;
    }
    case 10: {
      source_format = dnnl::memory::format_tag::abcdefghij;
      break;
    }
    case 11: {
      source_format = dnnl::memory::format_tag::abcdefghijk;
      break;
    }
    case 12: {
      source_format = dnnl::memory::format_tag::abcdefghijkl;
      break;
    }
    default: {
      source_format = dnnl::memory::format_tag::any;
      break;
    }
  }
  return source_format;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime
