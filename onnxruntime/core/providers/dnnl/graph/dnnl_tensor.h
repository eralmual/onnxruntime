// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "dnnl.hpp"
#include "core/providers/dnnl/graph/dnnl_onnx_tensor.h"

namespace onnxruntime::ort_dnnl {

// Copy of dnnl_primitive_config.h
using DnnlInputs = std::unordered_map<int, dnnl::memory>;
using DnnlPrimitive = std::pair<dnnl::primitive, DnnlInputs>;
using DnnlPrimitiveVector = std::unique_ptr<std::vector<DnnlPrimitive>>;

static const std::unordered_map<size_t, dnnl::memory::format_tag> FormatMap = 
{
    {0, dnnl::memory::format_tag::any},
    {1, dnnl::memory::format_tag::x},
    {2, dnnl::memory::format_tag::nc},
    {3, dnnl::memory::format_tag::ncw},
    {4, dnnl::memory::format_tag::nchw},
    {5, dnnl::memory::format_tag::ncdhw},
    {6, dnnl::memory::format_tag::abcdef},
    {7, dnnl::memory::format_tag::abcdefg},
    {8, dnnl::memory::format_tag::abcdefgh},
    {9, dnnl::memory::format_tag::abcdefghi},
    {10, dnnl::memory::format_tag::abcdefghij},
    {11, dnnl::memory::format_tag::abcdefghijk},
    {12, dnnl::memory::format_tag::abcdefghijkl},
};

class New_DnnlTensor {
 public:
  // Constructors
  // Constructor used for tensors based off onnx graph, used for graph build
  New_DnnlTensor(const NodeArg* arg, bool isConstantInitializer);
  // Constructor for artificial tensors used when decomposing complex ops
  New_DnnlTensor(New_DnnlTensor& base_tensor, const std::string& tensor_name, bool isConstantInitializer);
  New_DnnlTensor( const dnnl::memory::dims& dims, const dnnl::memory::data_type& dt, 
                  const std::string& tensor_name, bool isConstantInitializer);
  New_DnnlTensor();
  // Tensor info
  inline const std::string& Name()            { return name_; }
  inline const dnnl::memory::dims& Dims()     { return dnnl_dims_; }
  inline const dnnl::memory::data_type& DataType()    { return dnnl_dt_; }
  inline const dnnl::memory::dims& Strides()  { return dnnl_strides_; }
  inline const dnnl::memory::desc& MemoryDesc() { return dnnl_md_; }
  inline const dnnl::memory& Memory() const   { return *dnnl_memory_; }
  // Generate an memory md with layout any, so the primitive can decide the best layout
  dnnl::memory::desc GenerateInputMemoryDesc() { return dnnl::memory::desc(dnnl_dims_, dnnl_dt_, dnnl::memory::format_tag::any); }
  // Generate an memory md with plain layout, safer for binary prims
  dnnl::memory::desc GeneratePlainMemoryDesc() { return dnnl::memory::desc(dnnl_dims_, dnnl_dt_, FormatMap.at(dnnl_dims_.size())); }
  // Set a specific memory object, updates tensor info
  void SetMemory(std::unique_ptr<dnnl::memory> dnnl_memory);
  // Set a specific data handle to the memory object, this helps with reshape handling
  void SetMemoryDataHandle(void* data_handle, dnnl::stream dnnl_stream);
  // Reorder the memory as the descriptor says
  void ReorderMemory(const dnnl::memory::desc& new_md, std::vector<DnnlPrimitive>* prim_list, const dnnl::engine* target_engine = nullptr);
  // Get data from inner data handle as a vector with the specified data type
  template<typename T> std::vector<T> GetDataAsVector();
  // Increase the number of consumers
  inline void Consume()               { ++consumers_; }
  // Get the number of consumers
  inline const size_t& NumConsumers() { return consumers_; }
  // Signal the tensor is a graph input
  inline void SetAsGraphInput()       { is_graph_input_ = true; }
  // Signal the tensor is a graph output
  inline void SetAsGraphOutput()      { is_graph_output_ = true; }
  // Make the tensor a scalar
  inline void SetAsScalar()           { is_scalar_ = true; }
  // Check whether the tensor is dynamic, e.g. contains unspecified dimension
  inline bool HasDynamicDims()        { return has_dynamic_dims_; }
  // Check whether the tensor exsits for optional input output
  inline bool Exists()                { return name_ != ""; }
  // Check whether the tensor is constant initializer
  inline bool IsConstantInitializer() { return is_constant_initializer_; }
  // Check whether the tensor is scalar (not in dimensions but as defined by ONNX)
  inline bool IsScalar()              { return is_scalar_; }
  // Used to evaluate if this tensor can be used for inplace operations
  inline bool IsGraphOutput()         { return is_graph_output_; }
  // Used to evaluate if this tensor can be used for inplace operations
  inline bool IsGraphInput() { return is_graph_input_; }
  // Reset internal default values without losing ONNX info
  void ResetTensorConfig();

 private:
  // Keep this order since we use intialization lists on costructor
  // Stores tensor data
  std::string name_;
  // Store ONNX tensor information
  std::unique_ptr<OnnxTensor> onnx_tensor_;
  // Tensor dims
  dnnl::memory::dims dnnl_dims_;
  // Tensor data type
  dnnl::memory::data_type dnnl_dt_;
  // Tensor strides
  dnnl::memory::dims dnnl_strides_;
  // Tensor memory descriptor
  dnnl::memory::desc dnnl_md_;
  // dnnl memory object where we store our data
  std::unique_ptr<dnnl::memory> dnnl_memory_;
  // List of old memories used to reorder for recomended configurations
  std::vector<std::unique_ptr<dnnl::memory>> old_memories_;
  // List reshapes, resolved when setting data handles
  std::vector<std::pair<dnnl::memory, dnnl::memory>> reshapes_;
  // True if the tensor is a ORT constant initializer, else false
  bool is_constant_initializer_ = false;
  // True if the tensor has dynamic dimensions, else false
  bool has_dynamic_dims_ = false;
  // True if the tensor is a ORT scalar, else false. (ORT scalar is not the same as a dnnl scalar)
  bool is_scalar_ = false;
  /* Used to evaluate if this tensor can be used for inplace operations, if true don't inplace since when setting 
   * data handles we would have either copy the input data to the output to prevent the output data hanlde overwrite 
   * the input this would imply overhead ***ASUMED*** greater than the improvement given by using inplace ops
  */
  bool is_graph_output_ = false;
  // If input then the set data handle functions is a bit differentVectorOfDnnlTensorRawPtr
  bool is_graph_input_ = false;
  // Keep count on the number of consumers used for fusion
  size_t consumers_ = 0;


  void SwapMemory(std::unique_ptr<dnnl::memory>& new_memory);

  inline void CheckDynamicDims();
  inline bool AreDimsEquivalent(const dnnl::memory::dims& new_shape);
  inline void UpdateInfo();
};

// Add this for redability
using DnnlTensorPtr = std::shared_ptr<New_DnnlTensor>;
using UniqueSharedDnnlTensorPtr = std::unique_ptr<std::shared_ptr<New_DnnlTensor>>;
// We might be deleting nodes so make sure I/O will exist
using DnnlTensorPtrVector = std::shared_ptr<std::vector<DnnlTensorPtr*>>;

// Utility function
inline UniqueSharedDnnlTensorPtr dnnl_make_tensor(const NodeArg* arg = nullptr, bool isConstantInitializer = false) {
  if (arg) {
    return std::make_unique<DnnlTensorPtr>(std::make_shared<New_DnnlTensor>(arg, isConstantInitializer));
  } else {
    return std::make_unique<DnnlTensorPtr>(std::make_shared<New_DnnlTensor>());
  }
}

inline UniqueSharedDnnlTensorPtr dnnl_make_tensor(New_DnnlTensor& base_tensor, const std::string& tensor_name, bool isConstantInitializer) {
  return std::make_unique<DnnlTensorPtr>(std::make_shared<New_DnnlTensor>(base_tensor, tensor_name, isConstantInitializer));
}

// We will use this node to represent empty inputs or output across all primitives
static const DnnlTensorPtr empty_tensor = std::make_shared<New_DnnlTensor>();

// Had to define this here beacuse of templates :(
template <typename T>
std::vector<T> New_DnnlTensor::GetDataAsVector() {
  // Get the number of elements in the memory
  size_t num_elements = 1;
  for (auto& dim : dnnl_dims_) {
    num_elements *= dim;
  }
  // Generate container 
  std::vector<T> vector_data;
  // Select the correct data cast
  switch (dnnl_dt_) {
    case dnnl::memory::data_type::f32:
    {
      auto data = static_cast<float*>(dnnl_memory_->get_data_handle());
      auto data_lim = data + num_elements;
      for (; data < data_lim; ++data) {
        vector_data.emplace_back(static_cast<T>(*data));
      }
    }
      break;
    case dnnl::memory::data_type::s32:
    {
        auto data = static_cast<int32_t*>(dnnl_memory_->get_data_handle());
      auto data_lim = data + num_elements;
      for (; data < data_lim; ++data) {
        vector_data.emplace_back(static_cast<T>(*data));
    }
      }
      break;
    case dnnl::memory::data_type::s8:
    {
      auto data = static_cast<int8_t*>(dnnl_memory_->get_data_handle());
      auto data_lim = data + num_elements;
      for (; data < data_lim; ++data) {
        vector_data.emplace_back(static_cast<T>(*data));
      }
    }
      break;
    case dnnl::memory::data_type::u8:
    {
      auto data = static_cast<uint8_t*>(dnnl_memory_->get_data_handle());
      auto data_lim = data + num_elements;
      for (; data < data_lim; ++data) {
        vector_data.emplace_back(static_cast<T>(*data));
      }
    }
      break;
    default:
      ORT_THROW("[oneDNN] Error: Tried to generate vector from data using unsupported cast");
  }
  return vector_data;
}

}  // namespace onnxruntime::ort_dnnl