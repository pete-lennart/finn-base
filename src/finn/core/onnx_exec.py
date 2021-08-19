# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import copy
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import onnx.helper as helper
import onnxruntime as rt

import finn.analysis.topology as ta
import finn.core.execute_custom_node as ex_cu_node
from finn.core.modelwrapper import ModelWrapper
from finn.core.remote_pynq_exec import remote_pynq_exec
from finn.core.local_pynq_exec import local_pynq_exec
from finn.core.rtlsim_exec import rtlsim_exec
from finn.custom_op.registry import getCustomOp
from finn.util.basic import (
    get_sanitize_quant_tensors,
    is_finn_op,
    sanitize_quant_values,
)
from finn.util.invariant import invariant


Context = Dict[str, np.ndarray]
ONNXNode = Any
ONNXGraph = Any
ComparisonFN = Callable[[List[float], List[float]], bool]


def _translate_context(
    context: Context, from_names: List[str], to_names: List[str]
) -> None:
    """Translates the names given by 'from_names' in the context into names
    given by 'to_names'."""

    for (from_name, to_name) in zip(from_names, to_names):
        if from_name != to_name:
            context[to_name] = context[from_name]
            del context[from_name]


def _shift_context(
    context: Context,
    partition_node: ONNXNode,
    partition_model: ModelWrapper,
    shift_down: bool,
) -> None:
    """Shifts a context from a parent model with embedding partition node to
    context of the partition model. Since inputs and outputs of the embedded
    partition node and the partition model may not be the same, the former is
    translated to the latter."""

    partition_model_io = partition_model.graph.input + partition_model.graph.output
    partition_names = [x.name for x in partition_model_io]
    global_names = partition_node.input + partition_node.output
    if shift_down:
        _translate_context(context, global_names, partition_names)
    else:
        _translate_context(context, partition_names, global_names)


def _fork_partition_context(
    parent_context: Context, partition_node: ONNXNode, partition_model: ModelWrapper
) -> Context:
    """Create an execution context for the partition node / model to be used on
    executing the node / model. The inputs for the partition model are taken
    from the inputs of the node in the embedding model."""

    partition_context = dict(
        filter(lambda x: x[0] in partition_node.input, parent_context.items())
    )
    _shift_context(partition_context, partition_node, partition_model, True)
    return partition_context


def _reintegrate_partition_context(
    parent_context: Context,
    partition_context: Context,
    partition_node: ONNXNode,
    partition_model: ModelWrapper,
) -> None:
    """Reintegrate the partition context into the parent context."""

    partition_model_outputs = [x.name for x in partition_model.graph.output]
    partition_node_outputs = partition_node.output
    for name in partition_context.keys():
        if name not in partition_model_outputs:
            parent_context[partition_node.name + "_" + name] = partition_context[name]
    for (node_output, model_output) in zip(
        partition_node_outputs, partition_model_outputs
    ):
        parent_context[node_output] = partition_context[model_output]


def _execute_partition_node(node: ONNXNode, context: Context) -> None:
    """Executes a partition node. This can either be a generic partition node or
    a streaming dataflow partition node."""

    partition_node = getCustomOp(node)
    partition_model = ModelWrapper(partition_node.get_nodeattr("model"))
    partition_context = _fork_partition_context(
        context, partition_node, partition_model
    )

    invariant(
        len(partition_model.graph.input) == len(node.input),
        "Streaming dataflow partition model and node must have same number of inputs",
    )
    invariant(
        len(partition_model.graph.output) == len(node.output),
        "Streaming dataflow partition model and node must have same number of outputs",
    )
    invariant(
        len(partition_context) == len(node.input),
        "Provided context is missing some inputs",
    )

    partition_context = execute_onnx(partition_model, partition_context, True)

    if partition_model.get_metadata_prop("exec_mode") == "rtlsim":
        partition_model.save(partition_node.get_nodeattr("model"))

    _reintegrate_partition_context(context, partition_context, node, partition_model)


def _execute_generic_partition_node(node: ONNXNode, context: Context) -> None:
    _execute_partition_node(node, context)


def _execute_streaming_dataflow_partition_node(
    node: ONNXNode, context: Context
) -> None:
    invariant(
        len(node.input) == 1,
        "Streaming dataflow partition nodes must have exactly one input",
    )
    invariant(
        len(node.output) == 1,
        "Streaming dataflow partition nodes must have exactly one output",
    )
    _execute_partition_node(node, context)


def _execute_node_with_onnx_runtime(
    node: ONNXNode, context: Context, graph: ONNXGraph
) -> None:
    # onnxruntime unfortunately does not implement run_node as defined by ONNX,
    # it can only execute entire models -- so we create a model which solely
    # consists of our current node.
    # note: ensure that the same ValueInfo does not appear both in
    # graph.value_info as well as graph.output or graph.input
    # nodes with multiple outputs that are a mix of value_info and
    # input/outputs may get them reordered below
    node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
    node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
    node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
    node_graph = helper.make_graph(
        nodes=[node],
        name="single-node-exec",
        inputs=node_inputs,
        outputs=node_outputs,
    )
    node_model = helper.make_model(node_graph)
    input_dict = dict()
    for inp in node.input:
        input_dict[inp] = context[inp]

    sess = rt.InferenceSession(node_model.SerializeToString())
    output_list = sess.run(None, input_dict)

    for output_ind in range(len(node.output)):
        # get the name of the target buffer from node.output
        outp = node.output[output_ind]

        # retrieve the index of that name in node_outputs
        for i in range(len(node_outputs)):
            if outp == node_outputs[i].name:
                list_ind = i

        # use that index to index output_list
        if output_list[list_ind].shape != context[outp].shape:
            raise Exception(
                """Output shapes disagree after node execution:
                found %s vs expected %s"""
                % (str(output_list[list_ind].shape), str(context[outp].shape))
            )
        context[outp] = output_list[list_ind]


def execute_node(node: ONNXNode, context: Context, graph: ONNXGraph) -> None:
    """Executes a single node by using onnxruntime, with custom function or if
    dataflow partition by using remote/local execution or rtlsim. Input/output
    provided via context."""

    if node.op_type == "GenericPartition":
        _execute_generic_partition_node(node, context)
    elif node.op_type == "StreamingDataflowPartition":
        _execute_streaming_dataflow_partition_node(node, context)
    elif is_finn_op(node.domain):
        ex_cu_node.execute_custom_node(node, context, graph)
    else:
        _execute_node_with_onnx_runtime(node, context, graph)


def _default_exec(
    model: ModelWrapper,
    execution_context: Context,
    start_node: Optional[str] = None,
    end_node: Optional[str] = None,
) -> None:
    # execute the model node by node
    # we can simply walk down the list since the ONNX spec guarantees that it is
    # topologically sorted
    graph = model.graph
    subgraph = []
    if start_node is None:
        start_node = model.graph.node[0]
    if end_node is None:
        end_node = model.graph.node[-1]
    # select the nodes between specified start/end nodes
    start_ind = model.get_node_index(start_node)
    end_ind = model.get_node_index(end_node) + 1
    assert end_ind >= start_ind, "Start/end nodes must define valid subgraph"
    subgraph = graph.node[start_ind:end_ind]
    for node in subgraph:
        if get_sanitize_quant_tensors() != 0:
            # round input values to match quantization annotation
            execution_context = sanitize_quant_values(
                model, node.input, execution_context
            )
        execute_node(node, execution_context, graph)
        if get_sanitize_quant_tensors() != 0:
            # round output values to quantization annotation
            execution_context = sanitize_quant_values(
                model, node.output, execution_context
            )


def _generate_execution_context(model: ModelWrapper, input_dict: Context) -> Context:
    """Generates a new execution context for execution of the given model.
    Inputs to the model are taken from the provided 'input_dict'."""

    execution_context = model.make_empty_exec_context()
    for inp_name in input_dict.keys():
        if inp_name in execution_context:
            invariant(
                execution_context[inp_name].shape == input_dict[inp_name].shape,
                f"""Shape mismatch for provided input {inp_name}:
                found {str(execution_context[inp_name].shape)},
                expected {str(input_dict[inp_name].shape)}""",
            )
            execution_context[inp_name] = input_dict[inp_name]
    return execution_context


def execute_onnx(
    model: ModelWrapper,
    input_dict: Context,
    return_full_exec_context: bool = False,
    start_node: Optional[str] = None,
    end_node: Optional[str] = None,
) -> Context:
    """Executes given ONNX ModelWrapper with given named inputs.

    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.

    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.

    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    """

    model_exec_mode = model.get_metadata_prop("exec_mode")
    allowed_exec_modes = ["remote_pynq", "local_pynq", "rtlsim", "", None]

    invariant(
        model.check_all_tensor_shapes_specified(),
        "Found unspecified tensor shapes, try infer_shapes",
    )
    invariant(
        model.analysis(ta.nodes_topologically_sorted)["nodes_topologically_sorted"],
        "Nodes must be topologically sorted.",
    )
    invariant(
        model_exec_mode in allowed_exec_modes,
        """Metadata property "exec_mode" is set to an unknown value. Can be left
        unset or has to be set to "remote_pynq" for remote execution on a PYNQ
        board, "local_pynq" for local execution on a PYNQ board or "rtlsim" for
        execution using pyverilator!""",
    )

    execution_context = _generate_execution_context(model, input_dict)

    if model_exec_mode == "remote_pynq":
        remote_pynq_exec(model, execution_context)
    elif model_exec_mode == "local_pynq":
        local_pynq_exec(model, execution_context)
    elif model_exec_mode == "rtlsim":
        rtlsim_exec(model, execution_context)
    else:
        _default_exec(model, execution_context, start_node, end_node)

    if return_full_exec_context:
        return execution_context

    output_dict = dict()
    for out_tensor in model.graph.output:
        out_name = out_tensor.name
        output_dict[out_name] = execution_context[out_name]
    return output_dict


def execute_onnx_and_make_model(
    model: ModelWrapper, input_dict: Context
) -> ModelWrapper:
    """Executes given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""

    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    for i in execution_context.keys():
        new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a: ModelWrapper,
    model_b: ModelWrapper,
    input_dict: Context,
    compare_fxn: ComparisonFN = lambda x, y: np.isclose(x, y, atol=1e-3).all(),
) -> bool:
    """Executes two ONNX models and compare their outputs using given
    function."""

    res_a = list(execute_onnx(model_a, input_dict).items())[0][1]
    res_b = list(execute_onnx(model_b, input_dict).items())[0][1]
    return compare_fxn(res_a, res_b)
