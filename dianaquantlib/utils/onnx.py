from functools import partial
from typing import List, NamedTuple, Optional, Tuple
from quantlib.backends.base.onnxannotator import ONNXAnnotator
import quantlib.editing.graphs as qg

import onnxruntime as rt

import torch
import numpy as np
from torch import nn
import os
import onnx


class DianaAnnotator(ONNXAnnotator):
    def __init__(self, backend_name: str):

        super(DianaAnnotator, self).__init__(backend_name)

    def _annotate(self, network: nn.Module, onnxproto: onnx.ModelProto):

        # Define backend-specific supported ONNX nodes. The nodes belonging to
        # different node classes will be annotated using a class-specific logic.
        dory_onnxnode_op_types = {
            "linear": {"Conv", "Gemm"},
            "mul": {"Mul"},
            "add": {"Add"},
            "clip": {"Clip"},
            "div": {"Div"},
        }

        for n in onnxproto.graph.node:

            op_type = n.op_type
            annotations = []

            if op_type in dory_onnxnode_op_types["linear"]:
                op_name = n.input[1].rsplit(".", 1)[0]
                pytorch_module = network.get_submodule(op_name)
                if isinstance(
                    pytorch_module,
                    (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d),
                ):
                    weight_bits = (
                        2
                        if pytorch_module.is_analog == torch.Tensor([1])
                        else 8
                    )  # check register buffer inside conv layer to know if it's analog or not
                    bias_bits = 32
                    annotations.append(
                        onnx.helper.make_attribute(
                            key="weight_bits", value=weight_bits
                        )
                    )
                    annotations.append(
                        onnx.helper.make_attribute(
                            key="bias_bits", value=bias_bits
                        )
                    )
                    if isinstance(pytorch_module, nn.Conv2d):
                        annotations.append(
                            onnx.helper.make_attribute(
                                key="weight_data_layout", value="OIHW"
                            )
                        )
                        annotations.append(
                            onnx.helper.make_attribute(
                                key="input_data_layout", value="NCHW"
                            )
                        )

            elif (
                op_type in dory_onnxnode_op_types["mul"]
            ):  # only for analog core. still don't have specs
                mul_bits = 8
                annotations.append(
                    onnx.helper.make_attribute(key="mult_bits", value=mul_bits)
                )

            elif op_type in dory_onnxnode_op_types["add"]:
                add_bits = 8
                annotations.append(
                    onnx.helper.make_attribute(key="add_bits", value=add_bits)
                )

            else:  # the backend does not require special handling for this node type
                pass

            # flush attributes to the ONNX node
            n.attribute.extend(annotations)


class DianaExporter:
    """ONNX exporter class
    """

    def __init__(self):
        self.annotator = DianaAnnotator("DORY")
        # TODO: for now we assume a single input and single output
        self.input_names = ['input']
        self.output_names = ['output']

    def export(
        self,
        network: nn.Module,
        input_shape: Tuple,
        path: os.PathLike,
        name: Optional[str] = None,
        opset_version: int = 10,
    ) -> None:

        onnxname = name if name is not None else network._get_name()
        onnxfilename = (
            onnxname + "_QL_NOANNOTATION.onnx"
        )
        onnxfilepath = os.path.join(path, onnxfilename)

        # export the network (https://pytorch.org/docs/master/onnx.html#torch.onnx.export)
        torch.onnx.export(
            network,
            torch.randn(input_shape),  # a dummy input to trace the `nn.Module`
            onnxfilepath,
            input_names=self.input_names,
            output_names=self.output_names,
            export_params=True,
            do_constant_folding=True,
            opset_version=opset_version,
        )

        options = rt.SessionOptions()

        # Set graph optimization level
        options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        # To enable model serialization after graph optimization set this
        options.optimized_model_filepath = onnxfilepath
        # generate optimized graph from non-annotated model

        session = rt.InferenceSession(
            onnxfilepath,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        ## read optimized graph and annotate it with backend-specific information
        self.annotator.annotate(network, onnxfilepath)

    def dump_features(self,
                      network: nn.Module,
                      x:       torch.Tensor,
                      path:    os.PathLike) -> None:
        """Given a network, export the features associated with a given input.
        """

        class Features(NamedTuple):
            module_name: str
            features:    torch.Tensor

        # Since PyTorch uses dynamic graphs, we don't have symbolic handles
        # over the inner array. Therefore, we use PyTorch hooks to dump the
        # outputs of Requant layers.
        features: List[Features] = []

        def hook_fn(self, in_: torch.Tensor, out_: torch.Tensor, module_name: str):
            features.append(Features(module_name=module_name, features=out_))

        # 1. set up hooks to intercept features
        for n, m in network.named_modules():
            if isinstance(m, qg.nn.Requantisation):
                hook = partial(hook_fn, module_name=n)
                m.register_forward_hook(hook)

        # 2. propagate the supplied input through the network; the hooks will capture the features
        x = x.clone()
        y = network(x)

        # 3. export input, features, and output to .npy files
        np.save(os.path.join(path, f"{self.input_names[0]}.npy"), x.numpy())
        for i, (module_name, f) in enumerate(features):
            np.save(os.path.join(path, f"{module_name}_out{i}.npy"), f.detach().numpy())
        np.save(os.path.join(path, f"{self.output_names[0]}.npy"), y.detach().numpy())
