#!/usr/bin/env python

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import inspect
import time
from collections.abc import Iterable
from threading import Thread

import numpy as np
import simplejson as json
import smart_open
import torch
import torch.fx
import torch.onnx.symbolic_opset9
from torch import nn
BATCH_TYPE = Dict[str, Union[torch.Tensor, List[torch.Tensor], List[str], List[List[str]]]]

def _get_data_type(data):
    def check_iterable(d):
        return isinstance(d, Iterable) and callable(getattr(d, "__getitem__", None))

    if isinstance(data, torch.Tensor):
        return torch.Tensor
    elif check_iterable(data) and isinstance(data[0], torch.Tensor):
        return List[torch.Tensor]
    elif check_iterable(data) and isinstance(data[0], str):
        return List[str]
    elif check_iterable(data) and check_iterable(data[0]) and isinstance(data[0][0], str):
        return List[List[str]]
    else:
        return type(data)


def _split_batch_to_dict(
    data_dict: BATCH_TYPE,
):
    tensor_inputs: Dict[str, torch.Tensor] = dict()
    tensor_list_inputs: Dict[str, List[torch.Tensor]] = dict()
    str_inputs: Dict[str, List[str]] = dict()
    str_list_inputs: Dict[str, List[List[str]]] = dict()
    for feat, data in data_dict.items():
        data_type = _get_data_type(data)
        if data_type == torch.Tensor:
            tensor_inputs[feat] = data
        elif data_type == List[torch.Tensor]:
            tensor_list_inputs[feat] = data
        elif data_type == List[str]:
            str_inputs[feat] = data
        elif data_type == List[List[str]]:
            str_list_inputs[feat] = data
        else:
            raise ValueError(f'Unexpected data type {data_type} from feature "{feat}"')

    return_dict = dict()
    if tensor_inputs:
        return_dict["tensor_inputs"] = tensor_inputs
    if tensor_list_inputs:
        return_dict["tensor_list_inputs"] = tensor_list_inputs
    if str_inputs:
        return_dict["str_inputs"] = str_inputs
    if str_list_inputs:
        return_dict["str_list_inputs"] = str_list_inputs

    return return_dict


class TorchScriptDeployInfo(NamedTuple):
    """
    Information needed for deployment
    """

    input_names: List[str]
    output_names: List[str]

    # To determine shape. Current List is meant for multiple inputs. We may
    # need to iterate on this once we want to make sure model can support
    # variable input sizes (detection)
    sample_inputs: List[torch.Tensor]

    # how accurate the output needs to be
    convert_precision: Tuple[int, ...]

    # device to save. If None then don't force a specific device
    device: Optional[torch.device] = None


class TorchScriptEvalForwardDeployModel(nn.Module):
    """
    Wrapper of torch model to format it for deployment by calling eval_forward(*args, deploy=bool)
    """

    def __init__(self, model=None):
        super(TorchScriptEvalForwardDeployModel, self).__init__()
        self.model = model

    def forward(self, *args):
        outputs = self.model.eval_forward(*args, deploy=True)
        # Only tensor or tuple of tensors is supported by tracer.
        # Flatten the lists of tensors and convert to tuple of tensors.
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            return tuple(flatten(outputs))
        else:
            return outputs


def flatten(tuples):
    for x in tuples:
        if isinstance(x, tuple) or isinstance(x, list):
            yield from flatten(x)
        else:
            yield x


class TorchScriptTupleInputAdaptor(nn.Module):
    def __init__(self, model: nn.Module, input_order: List[str]):
        super().__init__()
        self.model = model
        self.input_order = input_order

    def forward(self, *tensors) -> Any:
        input_dict = dict(zip(self.input_order, tensors))
        return self.model(input_dict)


class MaybeConvertToDeviceInScope:
    """
    Context manager that converts a torch.nn.Module's parameter's device to `device` given only within
    the contextmanager scope. Will revert to the original model device upon exit
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        # both device and original_model_device must be specified for the device moving to actually happen
        self.device = device
        self.original_model_device = None

        if self.device is not None:
            # original device
            if hasattr(self.model, "device") and str(self.model.device) != "cuda":
                # if self.model.device is not default cuda device
                self.original_model_device = self.model.device
            else:
                try:
                    self.original_model_device = next(self.model.parameters()).device
                except StopIteration:
                    pass
                # There can be a failure case when buffers exist but no parameters.
                # If there's no parameters we just do not try to move any parameters/buffers to the new device.

            # we only support models with one device type for simplicity
            for parameter in self.model.parameters():
                assert parameter.device == self.original_model_device
            for buffer in self.model.buffers():
                assert buffer.device == self.original_model_device

    def __enter__(self):
        if self.device is not None and self.original_model_device is not None:
            self.model.to(self.device)

            if hasattr(self.model, "device"):
                self.model.device = self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None and self.original_model_device is not None:
            self.model.to(self.original_model_device)

            if hasattr(self.model, "device"):
                self.model.device = self.original_model_device


def benchmark_torch_script_model(model, inputs, warmup_iters=3, main_iters=10, optimized_execution=True):
    """
    Run the model several times, and measure the execution time.
    Return the execution time per iteration (millisecond).
    """
    for _i in range(warmup_iters):
        with torch.jit.optimized_execution(optimized_execution):
            model(*inputs)
    total_torch_script_time = 0.0
    for _i in range(main_iters):
        ts = time.time()
        with torch.jit.optimized_execution(optimized_execution):
            model(*inputs)
        te = time.time()
        total_torch_script_time += te - ts

    return total_torch_script_time * 1000 / main_iters


def compare_pytorch_vs_torch_script(
    inputs,
    pytorch_model,
    torch_script_module,
    outputs_expected_decimal,
    benchmark=True,
    torch_script_optimized_execution=True,
):
    """
    Check that pytorch and torchscript models output similar results
    """
    from caffe2.python.onnx.helper import benchmark_pytorch_model

    pytorch_results = pytorch_model(*inputs)

    if not isinstance(pytorch_results, tuple):
        pytorch_results = (pytorch_results,)

    assert len(outputs_expected_decimal) == len(pytorch_results), (
        "Need an expected decimal precision for each output. {} vs {}".format(
            len(outputs_expected_decimal), len(pytorch_results)
        )
    )

    with torch.jit.optimized_execution(torch_script_optimized_execution):
        torch_script_results = torch_script_module(*inputs)

    if not isinstance(torch_script_results, tuple):
        torch_script_results = (torch_script_results,)

    for p, t, expected_decimal in zip(pytorch_results, torch_script_results, outputs_expected_decimal):
        print("PyTorch results: {}".format(p.detach().cpu().numpy()))
        print("TorchScript results: {}".format(t.detach().cpu().numpy()))
        np.testing.assert_almost_equal(p.detach().cpu().numpy(), t.detach().cpu().numpy(), decimal=expected_decimal)
        print("Output achieves {}-decimal precision.".format(expected_decimal))

    if benchmark:
        pytorch_time = benchmark_pytorch_model(pytorch_model, inputs)
        torch_script_time = benchmark_torch_script_model(
            torch_script_module, inputs, optimized_execution=torch_script_optimized_execution
        )

        print(
            "PyTorch model's execution time is {} milliseconds/ iteration, {} iterations per second.".format(
                pytorch_time, 1000 / pytorch_time
            )
        )
        print(
            "TorchScript model's execution time is {} milliseconds / iteration, {} iterations per second".format(
                torch_script_time, 1000 / torch_script_time
            )
        )


def get_scriptable_pytorch_module_with_mixed_input_types(
    deploy_model: nn.Module,
    data_dict: BATCH_TYPE,
):
    """
    Converts PyTorch nn.Module to GraphModule given mixed input types, e.g. torch.Tensor and List[str].

    We expect deploy_model to implement these interfaces (do not add unused params):
    ```
    def forward(
        self,
        tensor_inputs: Dict[str, torch.Tensor],
        tensor_list_inputs: Dict[str, List[torch.Tensor]],
        str_inputs: Dict[str, List[str]],
        str_list_inputs: Dict[str, List[List[str]]],
    ) -> torch.Tensor:
    ```
    and
    ```
    def make_scriptable(
        self,
        tensor_inputs: Dict[str, torch.Tensor],
        tensor_list_inputs: Dict[str, List[torch.Tensor]],
        str_inputs: Dict[str, List[str]],
        str_list_inputs: Dict[str, List[List[str]]],
    ) -> nn.Module:
    ```

    :param deploy_model: PyTorch model (nn.Module)
    :param data_dict: sample data dict
    :return:
    """
    graph = torch.fx.Graph()
    make_scriptable_kwargs = _split_batch_to_dict(data_dict)
    deploy_model = deploy_model.make_scriptable(**make_scriptable_kwargs)

    result = graph.call_module(
        "model",
        kwargs={
            key: {
                feat: graph.placeholder("placeholder" + feat.replace("/", "_"), _get_data_type(data))
                for feat, data in value.items()
            }
            for key, value in make_scriptable_kwargs.items()
        },
    )
    graph.output(result)
    g = torch.fx.GraphModule(root={"model": deploy_model}, graph=graph)
    return g


def check_params_for_mixed_inputs(
    model,
    use_tuple_input_adaptor,
    scripting,
):
    if not scripting:
        raise ValueError(
            """Tracing is not supported for `forward` function with mixed input types,
            e.g. torch.Tensor and List[str].
            Please implement `make_scriptable` and apply `torch.jit.trace` to modules that require tracing.
            """
        )

    expected_parameters = {
        "tensor_inputs": "Dict[str, torch.Tensor]",
        "tensor_list_inputs": "Dict[str, List[torch.Tensor]]",
        "str_inputs": "Dict[str, List[str]]",
        "str_list_inputs": "Dict[str, List[List[str]]]",
    }

    forward_signature = inspect.signature(model.forward)
    unexpected_forward_parameters = []
    for parameter in forward_signature.parameters.values():
        annotation = expected_parameters.get(parameter.name)
        if annotation != parameter.annotation:
            unexpected_forward_parameters.append(parameter)
    has_make_scriptable = callable(getattr(model, "make_scriptable", None))
    unexpected_make_scriptable_parameters = []
    if has_make_scriptable:
        make_scriptable_signature = inspect.signature(model.make_scriptable)
        for parameter in make_scriptable_signature.parameters.values():
            annotation = expected_parameters.get(parameter.name)
            if annotation != parameter.annotation:
                unexpected_make_scriptable_parameters.append(parameter)

    if (
        unexpected_forward_parameters
        or not has_make_scriptable
        or unexpected_make_scriptable_parameters
        or use_tuple_input_adaptor
    ):
        error_message = """For `forward` function with mixed input types, e.g. torch.Tensor and List[str],
        we expect the class to implement the following interface (do not add unused parameters):
        ```
        def forward(
            self,
            tensor_inputs: Dict[str, torch.Tensor],
            tensor_list_inputs: Dict[str, List[torch.Tensor]],
            str_inputs: Dict[str, List[str]],
            str_list_inputs: Dict[str, List[List[str]]],
        ) -> torch.Tensor:
        ```
        and
        ```
        def make_scriptable(
            self,
            tensor_inputs: Dict[str, torch.Tensor],
            tensor_list_inputs: Dict[str, List[torch.Tensor]],
            str_inputs: Dict[str, List[str]],
            str_list_inputs: Dict[str, List[List[str]]],
        ) -> nn.Module:
        ```
        """

        if unexpected_forward_parameters:
            error_message = f"""{error_message}

        Unexpected parameters: {unexpected_forward_parameters}
        """

        if not has_make_scriptable:
            error_message = f"""{error_message}

        make_scriptable is not implemented
        """

        if unexpected_make_scriptable_parameters:
            error_message = f"""{error_message}

        Unexpected parameters: {unexpected_make_scriptable_parameters}
        """

        if use_tuple_input_adaptor:
            error_message = f"""{error_message}

        `use_tuple_input_adaptor` is not applicable in this case.
        """

        raise ValueError(error_message)


def pytorch_to_torch_script(
    deploy_info,
    model,
    out_path=None,
    benchmark=True,
    use_eval_forward=True,
    use_tuple_input_adaptor=False,
    scripting=False,
    module_info_filename="module_info.json",
    use_make_scriptable=False,
    artifact_path=None,
) -> None:
    """
    Convert pytorch model to torchscript

    Args:
        deploy_info: TorchScriptDeployInfo
        model: pytorch model (nn.Module)
        out_path: output path for the converted torchscript model
        benchmark: Whether to run a benchmarking for the pytorch vs torchscript converted models
        use_tuple_input_adaptor: Whether to wrap the module taking input as dict with tuple inout module
        use_eval_forward: Whether to use a Module's eval_forward method
        scripting: Whether to use scripting instead of tracing
        module_info_filename: output filename of extra file containing model metadata
        use_make_scriptable: whether to use module's make_scriptable method to generate scriptable model.
        Useful in cases where the model might not have mixed input but still requires specific scripting / tracing order.
        artifact_path: path to save model artifacts (e.g. module_info.json)
    """

    inputs = tuple(deploy_info.sample_inputs)
    has_mixed_inputs = any(
        _get_data_type(input_) == torch.Tensor or _get_data_type(input_) == List[torch.Tensor] for input_ in inputs
    ) and any(_get_data_type(input_) == List[str] or _get_data_type(input_) == List[List[str]] for input_ in inputs)

    if has_mixed_inputs:
        check_params_for_mixed_inputs(model, use_tuple_input_adaptor, scripting)

    # Move to device
    with MaybeConvertToDeviceInScope(model, deploy_info.device):
        if deploy_info.device is not None:
            # special case for text input
            def to_device(input_):
                data_type = _get_data_type(input_)
                if data_type == torch.Tensor:
                    return input_.to(deploy_info.device)
                elif data_type == List[torch.Tensor]:
                    return [i.to(deploy_info.device) for i in input_]
                else:
                    return input_

            inputs = [to_device(input_) for input_ in inputs]

        model_mode = model.training

        if use_eval_forward:
            deploy_model = TorchScriptEvalForwardDeployModel(model=model)
            if has_mixed_inputs or use_make_scriptable:
                deploy_model.make_scriptable = model.make_scriptable
        else:
            # Only reason we wrap above is to have forward(...) call eval_forward. not necessary if there's
            # no eval_forward
            deploy_model = model

        if use_tuple_input_adaptor:
            deploy_model = TorchScriptTupleInputAdaptor(deploy_model, input_order=deploy_info.input_names)

        deploy_model.train(mode=False)

        if has_mixed_inputs or use_make_scriptable:
            deploy_model = get_scriptable_pytorch_module_with_mixed_input_types(
                deploy_model, dict(zip(deploy_info.input_names, inputs))
            )

        print("Convert from pytorch to torchscript")
        if scripting:
            torch_script_module = torch.jit.script(deploy_model)
        else:
            with torch.jit.optimized_execution(True):
                torch_script_module = torch.jit.trace(deploy_model, inputs, check_trace=True)

        if out_path is not None:
            extra_files = {}
            extra_files[module_info_filename] = json.dumps(
                {"input_names": deploy_info.input_names, "output_names": deploy_info.output_names}
            )
            with smart_open.open(out_path, "wb") as f:
                torch.jit.save(torch_script_module, f, _extra_files=extra_files)

        print("Compare pytorch vs torchscript model")
        compare_pytorch_vs_torch_script(
            inputs, deploy_model, torch_script_module, deploy_info.convert_precision, benchmark=benchmark
        )

        model.train(mode=model_mode)


def pytorch_to_torch_script_in_thread(
    deploy_info: TorchScriptDeployInfo,
    model,
    out_prefix=None,
    benchmark=True,
    use_eval_forward=True,
    use_tuple_input_adaptor=False,
    scripting=False,
    module_info_filename="module_info.json",
    use_make_scriptable=False,
):
    """
    Run `pytorch_to_torch_script` in synchronous thread instead of inline. Previously used in caffe2 export
    as caffe2 had some environment cleanup issue that led to strange behavior. Unclear if needed now.
    """
    output_path = out_prefix + "_torch_script.pt"

    p = Thread(
        target=pytorch_to_torch_script,
        args=(deploy_info, model),
        kwargs=dict(
            out_path=output_path,
            benchmark=benchmark,
            use_eval_forward=use_eval_forward,
            use_tuple_input_adaptor=use_tuple_input_adaptor,
            scripting=scripting,
            module_info_filename=module_info_filename,
            use_make_scriptable=use_make_scriptable,
        ),
    )
    p.daemon = True
    p.start()
    p.join()

    return output_path


if __name__ == "__main__":
    pass
