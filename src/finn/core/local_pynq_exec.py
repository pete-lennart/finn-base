# Copyright (c) 2021 Xilinx, Inc.
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

import importlib.util
import os
from typing import Any, Dict, Tuple

from finn.util.invariant import invariant


def _import_accelerator_assets(deployment_directory: str) -> Tuple[Any, Dict[str, Any]]:
    driver_spec = importlib.util.spec_from_file_location(
        "driver", os.path.join(deployment_directory, "driver.py")
    )
    driver_base_spec = importlib.util.spec_from_file_location(
        "driver_base", os.path.join(deployment_directory, "driver_base.py")
    )
    driver_module = importlib.util.module_from_spec(driver_spec)
    driver_base_module = importlib.util.module_from_spec(driver_base_spec)
    driver_spec.loader.exec_module(driver_module)
    driver_base_spec.loader.exec_module(driver_base_module)
    return driver_base_module.FINNExampleOverlay, driver_module.io_shape_dict


def local_pynq_exec(model, execution_context):
    """Executes the given model locally on the pynq board. The metadata properties
    related to the pynq board have to be set. The execution context contains the
    input values."""

    platform = model.get_metadata_prop("platform")
    bitfile = model.get_metadata_prop("bitfile")
    deployment_directory = model.get_metadata_prop("pynq_deploy_dir")
    bitfile = os.path.basename(bitfile)
    runtime_weight_dir = "runtime_weights/"

    Accel, io_shape_dict = _import_accelerator_assets(deployment_directory)

    invariant(
        platform in ["alveo", "zynq-iodma"],
        "Platform must be either 'alveo' or 'zynq-iodma'",
    )

    input = execution_context[model.graph.input[0].name]
    batch_size = input.shape[0]
    accel = Accel(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=batch_size,
        runtime_weight_dir=runtime_weight_dir,
    )
    output = accel.execute(input)
    execution_context[model.graph.output[0].name] = output
