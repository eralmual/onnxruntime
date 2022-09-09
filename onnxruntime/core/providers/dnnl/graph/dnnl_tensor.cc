// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#include "dnnl_tensor.h"

namespace onnxruntime::ort_dnnl {

// Default values defines empty tensor characteristics
New_DnnlTensor::New_DnnlTensor() 
  : name_(""), dnnl_dims_({}), dnnl_dt_(dnnl::memory::data_type::undef) {}

New_DnnlTensor::New_DnnlTensor(const NodeArg* arg, bool isConstantInitializer)
    : name_((!arg || !arg->Exists()) ? "" : arg->Name()), // Get name if Onnx node is valid, else leave empty str
      onnx_tensor_(std::make_unique<OnnxTensor>(arg)),    // Generate the Onnx Tensor to handle data in ONNX format
      dnnl_dims_(onnx_tensor_->GetShapeInDnnlFormat()),   // Get original shape can be dynamic so can change at compile
      dnnl_dt_(onnx_tensor_->GetDataTypeInDnnlFormat()),  // Get original dt from the Onnx tensor info
      is_constant_initializer_(isConstantInitializer)     // Check if tensor is constant init for optimizations
{
  // Check for dynamic dims
  CheckDynamicDims();
}

New_DnnlTensor::New_DnnlTensor(New_DnnlTensor& base_tensor, const std::string& tensor_name, bool isConstantInitializer)
    : name_(tensor_name),
      dnnl_dims_(base_tensor.Dims()),
      dnnl_dt_(base_tensor.DataType()),
      is_constant_initializer_(isConstantInitializer) {
  // Check for dynamic dims
  CheckDynamicDims();
}

New_DnnlTensor::New_DnnlTensor(const dnnl::memory::dims& dims, const dnnl::memory::data_type& dt,
                               const std::string& tensor_name, bool isConstantInitializer)
    : name_(tensor_name),
      dnnl_dims_(dims),
      dnnl_dt_(dt),
      is_constant_initializer_(isConstantInitializer) {
  // Check for dynamic dims
  CheckDynamicDims();
}

void New_DnnlTensor::SwapMemory(std::unique_ptr<dnnl::memory>& new_memory) {
  // Set the memory 
  dnnl_memory_.swap(new_memory);
  // Update tensor info
  UpdateInfo();
}

void New_DnnlTensor::SetMemory(std::unique_ptr<dnnl::memory> dnnl_memory) {
  // Set the memory 
  dnnl_memory_ = std::move(dnnl_memory);
  // Update tensor info
  UpdateInfo();
}

void New_DnnlTensor::SetMemoryDataHandle(void* data_handle, dnnl::stream dnnl_stream) {
  // For graph inputs the first memory is the one where we start operating and the rest are intermediates,
  // whereas for outputs previous memories are intermediates and the last is the importan
  if (is_graph_input_ && !old_memories_.empty()) {
    old_memories_.front()->set_data_handle(data_handle, dnnl_stream);
  } else {
    dnnl_memory_->set_data_handle(data_handle, dnnl_stream);
  }  
  dnnl_stream.wait();
  // Propagate the pointer
  for (auto& reshape_pair : reshapes_) {
    reshape_pair.second.set_data_handle(reshape_pair.first.get_data_handle(), dnnl_stream);
    dnnl_stream.wait();
  }
}

void New_DnnlTensor::ReorderMemory( const dnnl::memory::desc& new_md, 
                                    std::vector<DnnlPrimitive>* prim_list, 
                                    const dnnl::engine* target_engine) {
  // In case we are passed the same md 
  if (dnnl_md_ == new_md) {
    return;
  }
  // Make sure descriptors are different and dimensions compatible
  auto new_dims = new_md.dims();
  // If dims are not compatible throw
  if (!AreDimsEquivalent(new_dims)) {
    ORT_THROW("[oneDNN] Error: Tried to reorder tensor memory with incompatible dimensions.");
  }

  // If no dnnl engine is provided use the current mem one
  dnnl::engine dnnl_engine; 
  if (target_engine) {
    dnnl_engine = *target_engine;
  } else {
    dnnl_engine = dnnl_memory_->get_engine();
  }

  // Create memory with the new descriptor
  auto new_mem = std::make_unique<dnnl::memory>(new_md, dnnl_engine);

  // This fuctions does a few things so we'll go step by step
  // 1) We reshape the memory if necesary
  if (new_dims != dnnl_dims_) {
    // Make a copy of the tensor mem but reshaped
    auto reshaped_original_md = dnnl_md_.reshape(new_md.dims());
    auto reshaped_mem = std::make_unique<dnnl::memory>(reshaped_original_md, dnnl_memory_->get_engine(), nullptr);
    // We handle things different for constant initializers
    if (IsConstantInitializer()) {
      // Generate a stream
      dnnl::stream dnnl_stream{dnnl_engine};
      // Set data handle and wait
      reshaped_mem->set_data_handle(dnnl_memory_->get_data_handle(), dnnl_stream);
      dnnl_stream.wait();
    } else {
      // Add as a reshape
      reshapes_.push_back({*dnnl_memory_, *reshaped_mem});
    }
    // Store memory
    SwapMemory(reshaped_mem);
    // reshaped_mem is now the old memory in our tensor, so store it
    old_memories_.emplace_back(std::move(reshaped_mem));
  }

  // 2) Now that we padded the memory, check if we need to cast of move to another engine
  if ((dnnl_md_ != new_md) || (dnnl_memory_->get_engine() != dnnl_engine)) {
    // For constant initializers we can do it inmediatly
    if (IsConstantInitializer()) {
      // Generate a stream and execute primitive
      dnnl::stream dnnl_stream{dnnl_engine};
      dnnl::reorder(*dnnl_memory_, *new_mem).execute(dnnl_stream, *dnnl_memory_, *new_mem);
      dnnl_stream.wait();
    } else {
      // For common tensors we generate the primitive bundle
      DnnlInputs io_map = {{DNNL_ARG_FROM, *dnnl_memory_}, {DNNL_ARG_TO, *new_mem}};
      auto reorder = dnnl::reorder(*dnnl_memory_, *new_mem);
      // Swap memory with current node memory
      SwapMemory(new_mem);
      // new_mem is now the old memory in our tensor, so store it
      old_memories_.emplace_back(std::move(new_mem));
      // Add the reorder to the list
      prim_list->emplace_back(std::make_pair(reorder, io_map));
    }
  }
}

void New_DnnlTensor::ResetTensorConfig() {
  if (!Exists()){
    return;
  }
  // Reset dnnl information
  dnnl_memory_.reset();
  // Delete old mems and reshapes
  old_memories_.clear();
  reshapes_.clear();
}

inline void New_DnnlTensor::CheckDynamicDims() {
  // Check for dynamic dims
  // If empty asume dynamic
  if (dnnl_dims_.size() == 0) {
    has_dynamic_dims_ = true;
  }
  // Evaluate each dim
  for (auto dim : dnnl_dims_) {
    if (dim == DNNL_RUNTIME_DIM_VAL) {
      has_dynamic_dims_ = true;
    }
  }
}

inline bool New_DnnlTensor::AreDimsEquivalent(const dnnl::memory::dims& new_shape) {
  return std::accumulate(std::begin(new_shape), std::end(new_shape), 
                 static_cast<dnnl::memory::dim>(1), std::multiplies<dnnl::memory::dim>()) ==
              std::accumulate(std::begin(dnnl_dims_), std::end(dnnl_dims_), 
                 static_cast<dnnl::memory::dim>(1), std::multiplies<dnnl::memory::dim>());
}

inline void New_DnnlTensor::UpdateInfo() {
  // Get memory info
  dnnl_md_ = dnnl_memory_->get_desc();
  auto strides = dnnl_md_.data.format_desc.blocking.strides;
  // Update class members
  dnnl_dims_ = dnnl_md_.dims();
  dnnl_dt_ = dnnl_md_.data_type();
  dnnl_strides_ = dnnl::memory::dims(strides, strides + dnnl_dims_.size());
}

}  // namespace onnxruntime::ort_dnnl