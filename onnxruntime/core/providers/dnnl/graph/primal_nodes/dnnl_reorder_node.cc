// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_reorder_node.h"

#include "core/providers/dnnl/graph/primal_nodes/dnnl_primitive_utils.h"

namespace onnxruntime {
namespace ort_dnnl {

std::unique_ptr<std::vector<DnnlPrimitive>> DnnlReorderNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config,
                                                                        const dnnl::engine& dnnl_engine) {
  // Get the input memory
  auto src_tensor = GetInput(IN_INPUT);
  // Get output tensor
  auto dst_tensor = GetOutput(OUT_OUTPUT);

  // Create container for dest_md
  dnnl::memory::desc dst_md;

  // Define the dst for cast operator and format tag reorder
  if (OpType() == "Cast") {
    GenerateCast(dst_md);
  } 

  // Add post ops
  dnnl::primitive_attr prim_attr;
  prim_attr.set_post_ops(post_op_config.post_ops);

  // Create the primitive descriptor
  auto reorder_pd = dnnl::reorder::primitive_desc(dnnl_engine, src_tensor->MemoryDesc(), dnnl_engine, dst_md);
  // Generate the memory
  auto dst_mem = std::make_unique<dnnl::memory>(reorder_pd.dst_desc(), dnnl_engine);
  // Create primititve
  auto reorder_prim = dnnl::reorder(reorder_pd);
  // Create the inputs
  post_op_config.input_map[DNNL_ARG_SRC] = src_tensor->Memory();
  post_op_config.input_map[DNNL_ARG_DST] = *dst_mem;

  // Set the output memory to the output tensor
  dst_tensor->SetMemory(std::move(dst_mem));
  // This is to comply with ONNX scalar handling, if sources are scalar then so is output
  if (src_tensor->IsScalar()) {
    dst_tensor->SetAsScalar();
  }

  // Create the primitive sequence object and append primitives
  auto primitive_seq = std::make_unique<std::vector<DnnlPrimitive>>();
  primitive_seq->emplace_back(std::make_pair(reorder_prim, post_op_config.input_map));

  // Create the Primitive bundle object
  return primitive_seq;
}

void DnnlReorderNode::GeneratePrimitive(DnnlPrimitiveConfig& post_op_config) {
  // Set the output type to the expected type
  post_op_config.src_out_type = GetOutput(OUT_OUTPUT)->DataType();
}

bool DnnlReorderNode::IsPostOpCompatible(DnnlPrimitiveConfig* post_op_config) {
  // If we are running the cast operator then evaluate
  if (OpType() == "Cast") {
    return IsCastPostOpCompatble(post_op_config);
  } else {
    // only cast is post op compatible for now
    return false;
  }
}

void DnnlReorderNode::BuildInPlace(DnnlPrimitiveConfig* post_op_config) {
  ORT_UNUSED_PARAMETER(post_op_config);
}

void DnnlReorderNode::CalculateOutputDims() {

  if (OpType() == "Cast") {
    // Set output dims eq to src dims
    output_dims_ = std::make_unique<dnnl::memory::dims>(GetInput(IN_INPUT)->Dims());
    // TODO: WE CAN OPTIMIZE UNSQUEEZE TO HANDLE ANY MEMORY LAYOUT SO WE DONT NEED THIS CHECK
    // If we are going to unsqueeze, make sure the memory is in the correct format
  } else if (OpType() == "Unsqueeze") {
    // TODO: PLACE HERE UNSQUEEZED DIMENTIONS
    output_dims_ = std::make_unique<dnnl::memory::dims>(GetInput(IN_INPUT)->Dims());
  }
}

inline bool DnnlReorderNode::IsCastPostOpCompatble(DnnlPrimitiveConfig* post_op_config) {
  auto reorder_dt = GetTo();
  // Check if the input - output combination is supported, if not, then don't fuse
  if (!primitive_utils::IsImplicitCastCompatible(post_op_config->src_primitive,
                                                 post_op_config->src_in_type,
                                                 reorder_dt)) {
    return false;
  }

  return true;
}

inline void DnnlReorderNode::GenerateCast(dnnl::memory::desc& dst_md) {
  // Get the src info to build the dst_md
  auto in_md = GetInput(IN_INPUT)->MemoryDesc();
  auto in_dims_size = in_md.dims().size();
  // Calculate the strides to avoid changing the memory layout
  auto strides = in_md.data.format_desc.blocking.strides;
  dnnl::memory::dims strides_vec(strides, strides + in_dims_size);
  // Generate dst md
  dst_md = dnnl::memory::desc(*output_dims_, GetTo(), strides_vec);
}

inline dnnl::memory::data_type DnnlReorderNode::GetTo() {
  // Get the attribute
  int64_t int_dt;
  auto atributes = GetAttributes();
  auto attr = atributes->find("to");
  if (attr != atributes->end()) {
    // When we create artificial attributes, we manually have to specify them as strings so take that into account
    if (attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
      int_dt = static_cast<int64_t>(std::stoi(attr->second().s()));
    } else {
      int_dt = attr->second().i();
    }
  } else {
    // to attribute should always exist in order to cast
    ORT_THROW("TO(CAST TARGET DATA TYPE) DOES NOT EXIST");
  }

  // Check fot the target datat ype
  if(int_dt == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return dnnl::memory::data_type::f32;
  } else if(int_dt == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return dnnl::memory::data_type::f16;
  } else if (int_dt == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    return dnnl::memory::data_type::bf16;
  } else if (int_dt == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    return dnnl::memory::data_type::s32;
  } else if (int_dt == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    return dnnl::memory::data_type::s8;
  } else if (int_dt == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    return dnnl::memory::data_type::u8;
  }else{
    ORT_THROW("Unsupported data type: ", int_dt);
  }
}

inline void DnnlReorderNode::GenerateUnsqueeze(const dnnl::engine& dnnl_engine) {
  ORT_UNUSED_PARAMETER(dnnl_engine);
  // Primitive container
  /*auto prep_prims = std::make_unique<std::vector<DnnlPrimitive>>();
  // Get I/O
  auto src_tensor = GetInput(IN_INPUT);
  auto axes_tensor = GetInput(IN_AXES);

  // TODO: WE CAN OPTIMIZE UNSQUEEZE TO HANDLE ANY MEMORY LAYOUT SO WE DONT NEED THIS CHECK
  // If we are going to unsqueeze, make sure the memory is in the correct format
  if (!primitive_utils::IsMemoryInExpectedOrtFormat(src_tensor->MemoryDesc())) {
    // Get dims
    auto in_dims = src_tensor->Dims();
    // Generate the memory in the correct format
    auto corrected_md = dnnl::memory::desc(in_dims, src_tensor->DataType(), FormatMap.at(in_dims.size()));
    auto corrected_mem = std::make_unique<dnnl::memory>(corrected_md, dnnl_engine);
    // Swap the memory with the tensor
    //src_tensor->SwapMemory(corrected_mem);
    // Create the primitive and set it into the list
    prep_prims->emplace_back(std::make_pair<dnnl::primitive, DnnlInputs>(
        dnnl::reorder(src_tensor->Memory(), *corrected_mem),
        {{DNNL_ARG_FROM, *corrected_mem},
         {DNNL_ARG_TO, src_tensor->Memory()}}));
    // Store the old mem
    aux_mem_.emplace_back(std::move(corrected_mem));
  }

  // The OneDNN execution provider automatically expands all scalar inputs to dim {1} tensors.
  // this will result in the data_dims.size() being 1 too large if the input is from a scalar.
  // To counter this data_dims is left empty if the input is from a scalar.
  dnnl::memory::dims data_dims;
  if (!src_tensor->IsScalar()) {
    data_dims = src_tensor->Dims();
  }

  std::vector<int64_t> axes_data;
  // ONNX Unsqueeze version 13+ the axes is an input tensor
  // ONNX Unsqueeze before version 13 axes comes from an Attribute.
  if (axes_tensor->Exists()) {
    dnnl::memory::dims axes_dims = axes_tensor->Dims();
    int64_t* p_axes_data = static_cast<int64_t*>(axes_tensor->Memory().get_data_handle());
    axes_data = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
  } else {
    axes_data = GetAxes();
  }

  // TODO: THIS COULD BE IMPROVED
  dnnl::memory::dims output_shape(axes_data.size() + data_dims.size(), 0);
  auto out_size = output_shape.size();
  // Set all axes indices to 1 in output_dims and check for duplicates
  for (int64_t axes : axes_data) {
    // Valid axis range is [0, output_rank - 1]
    axes = HandleNegativeAxis(axes, out_size);
    if (axes < 0 || axes >= static_cast<int64_t>(out_size))
      ORT_ENFORCE("'axes' has an out of range axis");
    if (output_shape[axes] != 0)
      ORT_ENFORCE("'axes' has a duplicate axis");
    output_shape[axes] = 1;
  }

  // Now fill in the zero entries with the existing shape
  {
    auto begin = data_dims.cbegin();
    for (auto& axisSize : output_shape) {
      if (axisSize == 0)
        axisSize = *begin++;
    }
    assert(begin == data_dims.cend());
  }
  // Set the new shape to the input tensor
  src_tensor->Reshape(output_shape);*/
  // nodes dont own the tensors, the graph does, nodes only get references 
  // to the tensors this way we can implement inplace
}

std::vector<int64_t> DnnlReorderNode::GetAxes() {
  // TODO: ADD SUPPORT FOR STRINGS SO WE CAN CREATE OUR OWN AXES
  auto atributes = GetAttributes();
  auto attr = atributes->find("axes");
  std::vector<int64_t> axes;
  if (attr != atributes->end() &&
      attr->second().type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS) {
    axes.reserve(attr->second().ints_size());
    for (int i = 0; i < attr->second().ints_size(); ++i) {
      axes.push_back(attr->second().ints(i));
    }
  } else {
    ORT_ENFORCE("Missing/Invalid 'axes' attribute value");
  }
  return axes;
}

}  // namespace ort_dnnl
}  // namespace onnxruntime