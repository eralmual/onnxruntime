// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <unordered_map>
#include <unordered_set>

#include "dnnl.hpp"

namespace onnxruntime::ort_dnnl::primitive_utils {

/* Used to generate oneDNN rules, usually key is the oneDNN primitive name and the vector
 *  contains the name of the supported primitives
 **/
using OneDnnRules = std::unordered_map<std::string, std::unordered_set<std::string>>;
// Used to determine if a oneDNN implicit cast is viable
using OneDnnCastRules = std::unordered_map<std::string,
                                           std::unordered_map<dnnl::memory::data_type,
                                                              std::unordered_set<dnnl::memory::data_type>>>;
// Used for 1:1 conversions between ONNX operator names to oneDNN primitive names
using OnnxOp2oneDnnPrimitive = std::unordered_map<std::string, std::string>;
// Used for 1:1 conversions between ONNX operator names to oneDNN algorithms
using OnnxOp2oneDnnAlgorithm = std::unordered_map<std::string, dnnl::algorithm>;
// Used to store ONNX operators that are executed with the same primitive
using OnePrimitive4All = std::unordered_set<std::string>;

static const OnePrimitive4All binary_ops = {"Add", "Div", "Equal", "Greater", "GreaterOrEqual",
                                            "Less", "LessOrEqual", "Mul", "Sub"};
static const OnePrimitive4All elementwise_ops = {"Abs", "Elu", "Exp", "LeakyRelu", "Log", "Relu",
                                                 "Round", "Sigmoid", "Softplus", "Sqrt", "Tanh"};
static const OnePrimitive4All pool_ops = {"AveragePool", "GlobalAveragePool", "GlobalMaxPool", "MaxPool"};
static const OnePrimitive4All reduce_ops = {"ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
                                            "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare"};

// Maps some ONNX spec operators to dnnl primitives 1:1
static const OnnxOp2oneDnnPrimitive onnx_op_to_onednn_primitive =
    {
        {"BatchNormalization", "BatchNormalization"},
        // Binary ops
        {"Add", "Binary"},
        {"Sub", "Binary"},
        {"Mul", "Binary"},
        {"Div", "Binary"},
        {"Equal", "Binary"},
        {"Greater", "Binary"},
        {"Less", "Binary"},
        {"GreaterOrEqual", "Binary"},
        {"LessOrEqual", "Binary"},
        // Cast op
        {"Cast", "Reorder"},
        // Conv ops
        {"Conv", "Convolution"},
        // Eltwise ops
        {"Abs", "Eltwise"},
        {"Elu", "Eltwise"},
        {"Exp", "Eltwise"},
        {"LeakyRelu", "Eltwise"},
        {"Log", "Eltwise"},
        {"Relu", "Eltwise"},
        {"Round", "Eltwise"},
        {"Sigmoid", "Eltwise"},
        {"Softplus", "Eltwise"},
        {"Sqrt", "Eltwise"},
        {"Tanh", "Eltwise"},
        {"Linear", "Eltwise"},
        {"Pow", "Eltwise"},
        // Gelu ops
        {"Gelu", "Eltwise"},
        {"FastGelu", "Eltwise"},
        // LRN op
        {"LRN", "LRN"},
        // Reduction ops
        {"LayerNormalization", "LayerNormalization"},
        // Matmul ops
        {"MatMul", "MatMul"},
        {"MatMulInteger", "MatMul"},
        // Pool ops
        {"AveragePool", "Pooling"},
        {"GlobalAveragePool", "Pooling"},
        {"GlobalMaxPool", "Pooling"},
        {"MaxPool", "Pooling"},
        // Reduction ops
        {"ReduceL1", "Reduction"},
        {"ReduceL2", "Reduction"},
        {"ReduceLogSum", "Reduction"},
        {"ReduceLogSumExp", "Reduction"},
        {"ReduceMax", "Reduction"},
        {"ReduceMean", "Reduction"},
        {"ReduceMin", "Reduction"},
        {"ReduceProd", "Reduction"},
        {"ReduceSum", "Reduction"},
        {"ReduceSumSquare", "Reduction"},
        // Reshape op
        {"Reshape", "Memory"},
        // Softmax op
        {"Softmax", "Softmax"},
        // Squeeze op
        {"Squeeze", "Memory"},
        // Sum op
        {"Sum", "Sum"},
        // Transpose op
        {"Transpose", "Reorder"},
        // Unsqueeze op
        {"Unsqueeze", "Memory"},
    };

// Maps algorithm name to dnnl::algorithm
static const OnnxOp2oneDnnAlgorithm onnx_op_to_onednn_algorithm =
    {
        // Binary algorithms
        {"Add",           dnnl::algorithm::binary_add},
        {"Mul",           dnnl::algorithm::binary_mul},
        {"Sub",           dnnl::algorithm::binary_sub},
        {"Div",           dnnl::algorithm::binary_div},
        // Eltwise algorithms
        {"Abs",           dnnl::algorithm::eltwise_abs},
        {"BiasGelu",      dnnl::algorithm::eltwise_gelu_erf},
        {"Elu",           dnnl::algorithm::eltwise_elu},        // Algorithm requires alpha value
        {"Equal",         dnnl::algorithm::binary_eq},
        {"Exp",           dnnl::algorithm::eltwise_exp},
        {"FastGelu",      dnnl::algorithm::eltwise_gelu_tanh},
        {"Gelu",          dnnl::algorithm::eltwise_gelu_erf},
        {"Greater",       dnnl::algorithm::binary_gt},
        {"GreaterOrEqual",dnnl::algorithm::binary_ge},
        {"LeakyRelu",     dnnl::algorithm::eltwise_relu},      // Algorithm requires alpha value
        {"Less",          dnnl::algorithm::binary_lt},
        {"LessOrEqual",   dnnl::algorithm::binary_le},
        {"Log",           dnnl::algorithm::eltwise_log},
        {"Pow",           dnnl::algorithm::eltwise_pow},
        {"Relu",          dnnl::algorithm::eltwise_relu},
        {"Round",         dnnl::algorithm::eltwise_round},
        // OneDNN eltwise_logistic is defined as 1/(1 + exp(-x)) which matches the definition of "Sigmoid" in ONNX
        {"Sigmoid",       dnnl::algorithm::eltwise_logistic},
        // OneDNN eltwise_soft_relu is defined as ln(1 + exp(x)) which matches the definition of "Softplus" in ONNX
        {"Softplus",      dnnl::algorithm::eltwise_soft_relu},
        {"Sqrt",          dnnl::algorithm::eltwise_sqrt},
        {"Tanh",          dnnl::algorithm::eltwise_tanh},
        {"Linear",        dnnl::algorithm::eltwise_linear}
    };

// Maps dnnl primitive names to the names of the post op compatible primitives
static const OneDnnRules onednn_postop_rules =
    {
        // Add reorder into ops that support implicit casts and format changes
        {"Convolution",         {"Eltwise", "Sum", "Binary", "Depthwise"}},
        {"InnerProduct",        {"Eltwise", "Sum", "Binary"}}, 
        {"MatMul",              {"Eltwise", "Sum", "Binary", "Reorder"}}, 
        {"BatchNormalization",  {"Eltwise"}},
        {"Binary",              {"Eltwise", "Sum", "Binary", "Reorder"}},
        {"Concat",              {}},
        {"Eltwise",             {"Binary"}},
        {"LayerNormalization",  {"Eltwise", "Sum", "Binary"}},
        {"LRN",                 {}},
        {"Pooling",             {}},
        {"PReLU",               {}},
        {"Resampling",          {"Eltwise", "Sum", "Binary"}},
        {"Shuffle",             {}},
        {"Softmax",             {}},
        {"Sum",                 {}},
        {"Reorder",             {"Sum"}},
        {"Reduction",           {"Eltwise", "Sum", "Binary"}}
    };

// Used for readability in dnnl data types on the map
using dnnl_dt = dnnl::memory::data_type;
// Maps dnnl primitive names to src0 and dst compatible data type
static const OneDnnCastRules onednn_src_dst_dt_rules =
    {
        {"Convolution", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::f16,  {dnnl_dt::f16, dnnl_dt::f32, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::u8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::f32, dnnl_dt::f16, dnnl_dt::bf16}},
            {dnnl_dt::s8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::f32, dnnl_dt::f16, dnnl_dt::bf16}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
        }},
        {"InnerProduct", {
            {dnnl_dt::f32,  {dnnl_dt::f32}},
            {dnnl_dt::f16,  {dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::u8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::f32}},
            {dnnl_dt::s8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::f32}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
        }},
        {"MatMul", {
            {dnnl_dt::f32,  {dnnl_dt::f32}},
            {dnnl_dt::f16,  {dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::u8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::s8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::s32, dnnl_dt::f32, dnnl_dt::bf16}},
        }},
        {"BatchNormalization", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::f16,  {dnnl_dt::f16}},
            {dnnl_dt::s8,   {dnnl_dt::s8}},
        }},
        {"Binary", {
            {dnnl_dt::bf16,  {dnnl_dt::bf16}},
            {dnnl_dt::s8,    {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
            {dnnl_dt::u8,    {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
            {dnnl_dt::f16,   {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
            {dnnl_dt::f32,   {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
        }},
        {"Concat", {}},     // Primitive is dt agnostic
        {"Eltwise", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::f16,  {dnnl_dt::f16}},
            {dnnl_dt::s32,  {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s8,   {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::u8,   {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
        }},
        {"LayerNormalization", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::f16,  {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::u8,   {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::s8,   {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::f16, dnnl_dt::u8, dnnl_dt::s8}},
        }},
        {"LRN", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::f16,  {dnnl_dt::f16}},
        }},
        {"Pooling", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::bf16, {dnnl_dt::bf16}},
            {dnnl_dt::f16,  {dnnl_dt::f16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s32,  {dnnl_dt::s32}},
            {dnnl_dt::s8,   {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
            {dnnl_dt::u8,   {dnnl_dt::s8, dnnl_dt::u8, dnnl_dt::f16, dnnl_dt::f32}},
        }},
        {"PReLU", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s32,  {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},    
            {dnnl_dt::s8,   {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::u8,   {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
        }},
        {"Resampling", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s32,  {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s8,   {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::u8,   {dnnl_dt::f32, dnnl_dt::s32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::f16,  {dnnl_dt::f16}},
        }},
        {"Shuffle", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::s32,  {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s8,   {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::u8,   {dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
        }},
        {"Softmax", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::u8, dnnl_dt::s8}},
            {dnnl_dt::f16,  {dnnl_dt::f16}},
            {dnnl_dt::u8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::f32, dnnl_dt::bf16}},
            {dnnl_dt::s8,   {dnnl_dt::u8, dnnl_dt::s8, dnnl_dt::f32, dnnl_dt::bf16}},
        }},
        {"Sum", {}},        // Works with every dt
        {"Reorder", {       // Works with every dt
#ifdef DNNL_CPU_RUNTIME
            // On CPU reorders between bf16 and s32 data types are not supported.
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::f16, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s32,  {dnnl_dt::f32, dnnl_dt::f16, dnnl_dt::s32, dnnl_dt::s8, dnnl_dt::u8}},
#endif
        }},
        {"Reduction", {
            {dnnl_dt::f32,  {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::bf16, {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::s8,   {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
            {dnnl_dt::u8,   {dnnl_dt::f32, dnnl_dt::bf16, dnnl_dt::s8, dnnl_dt::u8}},
        }},
    };

// Mapping functions
std::string OrtOperatorToDnnlPrimitiveName(const std::string& op);
dnnl::algorithm OrtOperatorToDnnlAlgorithm(const std::string& op);
bool IsSimpleOperator(const std::string& op);
bool IsPostOpCompatible(const std::string& source_op, const std::string& next_op);
bool IsImplicitCastCompatible(const std::string& primitive, const dnnl_dt& src_dt, const dnnl_dt& dst_dt);


// Padding functions
std::pair<dnnl::memory::desc, dnnl::memory::desc> PaddSourcesEqualy(const dnnl::memory::desc& src_1_md,
                                                                    const dnnl::memory::desc& src_2_md);
void PaddSourcesEqualy(dnnl::memory::dims& src_1_dims, dnnl::memory::dims& src_2_dims);
dnnl::memory::desc Padd(dnnl::memory::desc& target_md, size_t front_pad, size_t back_pad);

// Evaluates if the memory is in a fromat that ORT can use
bool IsMemoryInExpectedOrtFormat(const dnnl::memory::desc& desc);

}  // namespace onnxruntime::ort_dnnl::primitive_utils
