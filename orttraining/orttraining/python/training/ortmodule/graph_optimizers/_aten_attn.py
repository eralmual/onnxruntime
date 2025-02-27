# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
PyTorch's _efficient_attention_forward/_efficient_attention_backward APIs is keep changing. Current implementation
is tested well on version 2.3.0.dev20240221+cu118, and should be run well since official version 2.3.0. If may fail to
run is you are using PyTorch with older versions.

This file is more like an example of how to add a new graph optimizer. Ideally user can add graph optimizer according
to the specific model they are using on their own instead of putting every possible graph optimizer here.

PyTorch also has API for flash attention (currently doesn't support random attention mask or Dropout), we can add
support if we want to try in the future.
"""

from onnx import GraphProto, NodeProto, TensorProto, helper

from ..graph_optimizer_registry import register_graph_optimizer
from .utils import GraphMatcher, check_attribute_value, make_constant_node, update_graph


def _make_efficient_attention_nodes(
    idx: int,
    q: str,
    k: str,
    v: str,
    y: str,
    dy: str,
    dq: str,
    dk: str,
    dv: str,
    bias: str,
    expand_bias: bool,
    scale: float,
    dropout_ratio: float,
    causal: bool,
):
    nodes_to_add = []
    scale_node = make_constant_node("scale_" + str(idx), TensorProto.FLOAT, [], [scale])
    dropout_ratio_node = make_constant_node("dropout_ratio_" + str(idx), TensorProto.FLOAT, [], [dropout_ratio])
    causal_node = make_constant_node("causal_" + str(idx), TensorProto.INT64, [], [1 if causal else 0])
    one_node = make_constant_node("one_" + str(idx), TensorProto.INT64, [], [1])
    zero_node = make_constant_node("zero_" + str(idx), TensorProto.INT64, [], [0])
    logsumexp = helper.make_tensor_value_info("logsumexp" + str(idx), TensorProto.FLOAT, [])
    seed = helper.make_tensor_value_info("seed" + str(idx), TensorProto.INT64, [])
    offset = helper.make_tensor_value_info("offset" + str(idx), TensorProto.INT64, [])
    msb_q = helper.make_tensor_value_info("msb_q_" + str(idx), TensorProto.INT64, [])
    msb_k = helper.make_tensor_value_info("msb_k_" + str(idx), TensorProto.INT64, [])
    new_value_infos = [logsumexp, seed, offset, msb_q, msb_k]
    if expand_bias:
        shape_0 = helper.make_node("Shape", [q], ["shape_0_" + str(idx)], start=0, end=1)
        shape_1 = helper.make_node("Shape", [q], ["shape_1_" + str(idx)], start=2, end=3)
        shape_2 = helper.make_node("Shape", [q], ["shape_2_" + str(idx)], start=1, end=2)
        shape_3 = helper.make_node("Shape", [k], ["shape_3_" + str(idx)], start=1, end=2)
        concat = helper.make_node(
            "Concat",
            [shape_0.output[0], shape_1.output[0], shape_2.output[0], shape_3.output[0]],
            ["concated_shape_" + str(idx)],
            axis=0,
        )
        expand = helper.make_node("Expand", [bias, concat.output[0]], ["expanded_bias_" + str(idx)])
        nodes_to_add.extend([shape_0, shape_1, shape_2, shape_3, concat, expand])
        bias = expand.output[0]
    fwd_node = helper.make_node(
        "ATen",
        [
            q,
            k,
            v,
            bias,
            "",
            "",
            "",
            "",
            dropout_ratio_node.output[0],
            causal_node.output[0],
            one_node.output[0],
            scale_node.output[0],
            "",
            "",
        ],
        [y, logsumexp.name, seed.name, offset.name, msb_q.name, msb_k.name],
        "efficient_attention_forward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_forward",
        cpu_input_args=[4, 5, 12, 13],
        cpu_output_args=[2, 3, 4, 5],
    )
    bwd_node = helper.make_node(
        "ATen",
        [
            dy,
            q,
            k,
            v,
            bias,
            y,
            "",
            "",
            msb_q.name,
            msb_k.name,
            logsumexp.name,
            dropout_ratio_node.output[0],
            seed.name,
            offset.name,
            causal_node.output[0],
            zero_node.output[0],
            scale_node.output[0],
            "",
        ],
        [dq, dk, dv, ""],
        "efficient_attention_backward_" + str(idx),
        None,
        "org.pytorch.aten",
        operator="_efficient_attention_backward",
        cpu_input_args=[6, 7, 12, 13],
    )
    nodes_to_add.extend([scale_node, dropout_ratio_node, causal_node, one_node, zero_node, fwd_node, bwd_node])
    return nodes_to_add, new_value_infos


# Without causal mask, with Dropout. For example, BERT model in HuggingFace.
_PATTERN_0: list[tuple[str, bool, list[tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Transpose", True, [(0, 0, 1)]),  # 2
    ("Div", False, [(0, 0, 0)]),  # 3
    ("Add", False, [(3, 0, 0)]),  # 4
    ("Softmax", False, [(4, 0, 0)]),  # 5
    ("Dropout", False, [(5, 0, 0)]),  # 6
    ("MatMul", False, [(6, 0, 0)]),  # 7
    ("Transpose", True, [(7, 0, 1)]),  # 8
    ("Transpose", False, [(7, 0, 0)]),  # 9
    ("FusedMatMul", False, [(8, 0, 1)]),  # 10
    ("DropoutGrad", False, [(10, 0, 0), (6, 1, 1)]),  # 11
    ("SoftmaxGrad_13", False, [(11, 0, 0), (5, 0, 1)]),  # 12
    ("Identity", False, [(12, 0, 0)]),  # 13
    ("Div", False, [(13, 0, 0)]),  # 14
    ("Identity", False, [(14, 0, 0)]),  # 15
    ("FusedMatMul", False, [(2, 0, 1), (15, 0, 0)]),  # 16
    ("FusedMatMul", False, [(1, 0, 0), (15, 0, 1)]),  # 17
    ("FusedMatMul", False, [(6, 0, 0)]),  # 18
    ("Transpose", True, [(18, 0, 1)]),  # 19
    ("Transpose", False, [(16, 0, 0)]),  # 20
    ("Transpose", False, [(17, 0, 0)]),  # 21
    ("Transpose", False, [(18, 0, 0)]),  # 22
]


def _optimize_for_pattern_0(matcher: GraphMatcher, idx: int, nodes: list[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[3].input[1])
    ratio_value = matcher.get_constant_value(nodes[6].input[1])
    if not (
        check_attribute_value(nodes[1], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[2], "perm", [0, 2, 3, 1])
        and scale_value is not None
        and ratio_value is not None
        and check_attribute_value(nodes[8], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[9], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    _, add_input_shape_0 = matcher.get_type_and_shape(nodes[4].input[0])
    _, add_input_shape_1 = matcher.get_type_and_shape(nodes[4].input[1])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[1].input[0],
        nodes[2].input[0],
        nodes[8].input[0],
        nodes[9].output[0],
        nodes[19].input[0],
        nodes[20].output[0],
        nodes[21].output[0],
        nodes[22].output[0],
        nodes[4].input[1],
        add_input_shape_0 != add_input_shape_1,
        1 / float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        ratio_value,
        False,
    )
    return nodes, nodes_to_add, new_value_infos


# Without causal mask, without Dropout. For example, BERT model and disabling attention dropout in HuggingFace.
_PATTERN_1: list[tuple[str, bool, list[tuple[int, int, int]]]] = [
    ("MatMul", False, []),  # 0
    ("Transpose", True, [(0, 0, 0)]),  # 1
    ("Transpose", True, [(0, 0, 1)]),  # 2
    ("Div", False, [(0, 0, 0)]),  # 3
    ("Add", False, [(3, 0, 0)]),  # 4
    ("Softmax", False, [(4, 0, 0)]),  # 5
    ("MatMul", False, [(5, 0, 0)]),  # 6
    ("Transpose", True, [(6, 0, 1)]),  # 7
    ("Transpose", False, [(6, 0, 0)]),  # 8
    ("FusedMatMul", False, [(7, 0, 1)]),  # 9
    ("SoftmaxGrad_13", False, [(9, 0, 0), (5, 0, 1)]),  # 10
    ("Identity", False, [(10, 0, 0)]),  # 11
    ("Div", False, [(11, 0, 0)]),  # 12
    ("Identity", False, [(12, 0, 0)]),  # 13
    ("FusedMatMul", False, [(2, 0, 1), (13, 0, 0)]),  # 14
    ("FusedMatMul", False, [(1, 0, 0), (13, 0, 1)]),  # 15
    ("FusedMatMul", False, [(5, 0, 0)]),  # 16
    ("Transpose", True, [(16, 0, 1)]),  # 17
    ("Transpose", False, [(14, 0, 0)]),  # 18
    ("Transpose", False, [(15, 0, 0)]),  # 19
    ("Transpose", False, [(16, 0, 0)]),  # 20
]


def _optimize_for_pattern_1(matcher: GraphMatcher, idx: int, nodes: list[NodeProto]):
    # Check forward only as the backward is expected to be consistent if it's built correctly.
    scale_value = matcher.get_constant_value(nodes[3].input[1])
    if not (
        check_attribute_value(nodes[1], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[2], "perm", [0, 2, 3, 1])
        and scale_value is not None
        and check_attribute_value(nodes[7], "perm", [0, 2, 1, 3])
        and check_attribute_value(nodes[8], "perm", [0, 2, 1, 3])
    ):
        return [], [], []

    _, add_input_shape_0 = matcher.get_type_and_shape(nodes[4].input[0])
    _, add_input_shape_1 = matcher.get_type_and_shape(nodes[4].input[1])
    nodes_to_add, new_value_infos = _make_efficient_attention_nodes(
        idx,
        nodes[1].input[0],
        nodes[2].input[0],
        nodes[7].input[0],
        nodes[8].output[0],
        nodes[17].input[0],
        nodes[18].output[0],
        nodes[19].output[0],
        nodes[20].output[0],
        nodes[4].input[1],
        add_input_shape_0 != add_input_shape_1,
        1 / float(scale_value[0] if isinstance(scale_value, list) else scale_value),
        0.0,
        False,
    )
    return nodes, nodes_to_add, new_value_infos


_PATTERNS = [
    (_PATTERN_0, _optimize_for_pattern_0),
    (_PATTERN_1, _optimize_for_pattern_1),
]


@register_graph_optimizer(devices="cuda")
def optimize_graph_for_aten_efficient_attention(graph: GraphProto):
    nodes_to_remove = []
    nodes_to_add = []
    new_value_infos = []
    matcher = GraphMatcher(graph)
    idx = 0
    for pattern_tuple in _PATTERNS:
        for nodes in matcher.match_pattern(pattern_tuple[0]):
            remove_nodes, add_nodes, add_value_infos = pattern_tuple[1](matcher, idx, nodes)
            if len(add_nodes) > 0:
                nodes_to_remove.extend(remove_nodes)
                nodes_to_add.extend(add_nodes)
                new_value_infos.extend(add_value_infos)
                idx += 1
    update_graph(graph, nodes_to_remove, nodes_to_add, new_value_infos)
