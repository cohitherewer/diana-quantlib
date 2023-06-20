# Abstract class which all diana specific module have to inhereit from
from abc import abstractmethod
from cProfile import run
from collections import OrderedDict
from math import log2, floor
from pathlib import Path
from random import randint
import sched
from typing import Any, Dict, List, Union, Generator

from DianaModules.core.Operations import (
    AnalogConv2d,
    AnalogOutIdentity,
    DIANAConv2d,
    DIANAIdentity,
    DIANALinear,
    DIANAReLU,
    DianaBaseOperation,
)
from DianaModules.utils.converters.fake2true import LayerIntegrizationConverter
from DianaModules.utils.converters.float2fake import F2FConverter
from DianaModules.utils.converters.hwquantization import HWMappingConverter
from DianaModules.utils.onnx import DianaExporter
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qmodules import (
    _PACTActivation,
)
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd

from quantlib.algorithms.qmodules.qmodules.qmodules import (
    _QActivation,
    _QModule,
)
import torch
import torch.fx as fx
from torch import nn, rand
import quantlib.editing.graphs as qg
import quantlib.backends as qb
from torch import optim
import torch.utils.data as ut
from torch.utils.data import Dataset as ds


class DianaModule(nn.Module):  # Base class for all diana models
    def __init__(
        self,
        graph_module: Union[nn.Module, fx.graph_module.GraphModule],
        representative_dataset: Generator,
        dataset_scale: torch.tensor = torch.tensor(1.0)

    ):
        """
        graph_module:           floating-point pytorch module or graph module
        representative_dataset: generator function for obtaining a dataset for
                                estimating the quantiser parameters.
        dataset_scale:          This needs to be depricated since it should be
                                always 1.0.
                                TODO: we need to implement an option to
                                choose between float or int8 input/output.
                                During ONNX export this option should:
                                * If int8 input is chosen: remove the input DigitalRequantizer module
                                * If int8 output is chosen: remove the output EpsTunnel module
        """
        super().__init__()
        self.gmodule = graph_module
        self.representative_dataset = representative_dataset
        self.dataset_scale = dataset_scale
        self._integrized = False

    def start_observing(
        self, is_modules=(_QModule), not_modules=type(None)
    ):  # before starting training with FP
        for _, module in self.gmodule.named_modules():
            if isinstance(module, is_modules) and not isinstance(
                module, not_modules
            ):
                module.start_observing()

    def stop_observing(
        self, is_modules=(_QModule), not_modules=type(None)
    ):  # before starting training with fake quantised network
        for _, module in self.gmodule.named_modules():
            if isinstance(module, is_modules) and not isinstance(
                module, not_modules
            ):
                module.stop_observing()

    def forward(self, x: torch.Tensor):
        return self.gmodule(x)

    def freeze_clipping_bound(self):
        for _, module in self.gmodule.named_modules():
            if isinstance(module, DIANAReLU):
                module.freeze()

    def unfreeze_clipping_bound(self) -> None:
        for _, module in self.gmodule.named_modules():
            if isinstance(module, DIANAReLU):
                module.thaw()

    def convert_scales_to_pot(self):
        """Convert quantization scales to power-of-two scales
        """
        # round scales to power-of-two scales
        for module in self.gmodule.modules():
            if (isinstance(module, DIANAConv2d) or isinstance(module, DIANALinear)
                or (isinstance(module, _QActivation) and not isinstance(module, AnalogOutIdentity))):
                # if the number of levels is high enough, we prefer range over step size and use ceil instead of round
                round_op = torch.round if module.n_levels < 128 else torch.ceil
                module.scale = torch.Tensor(torch.exp2(round_op(torch.log2(module.scale))))

    def align_ews_input_scales(self):
        """Ensure that the quantization scales of both inputs of an element-wise-add are the same.
        This is needed since the digital add op cannot scale both inputs individually.
        """
        for name, module in self.gmodule.named_modules():
            # we match by name since harmonized add modules is traced as an nn.Module
            if 'AddTreeHarmoniser' not in name or '_input_qmodules' not in name:
                continue

            # TODO: if only one item in scales due to skip without ops, ensure that the scale is the
            # same as the dominating op (ReLU in ResNet).
            # -> make a rewriter transform for this
            scales = []
            for qm in module.modules():
                if isinstance(qm, DIANAIdentity):
                    scales.append(qm.scale)

            new_scale = torch.concat(scales).max().unsqueeze(0)
            for qm in module.modules():
                if isinstance(qm, DIANAIdentity):
                    qm.scale = new_scale

    @classmethod
    def from_trainedfp_model(
        cls, model: nn.Module, modules_descriptors=None,
        qhparamsinitstrategy: str = 'meanstd'
    ):  # returns fake quantized model from_ floating point quantised model
        # WARNING:
        #   modules_descriptors from the .yaml file are only used for the analog core currently
        #   (to see core type and quantization specs).
        #   All other layers are constructed according ot the modulewisedescriptionspec.
        #   TODO: we need to make this more convenient!!!
        modulewisedescriptionspec = (  # change const later
            (
                {"types": ("Identity")},
                (
                    "per-array",
                    {"bitwidth": 8, "signed": True},
                    qhparamsinitstrategy,
                    "DIANA",
                ),
            ),
            (
                {"types": ("ReLU")},
                (
                    "per-array",
                    {"bitwidth": 7, "signed": False},
                    qhparamsinitstrategy,
                    "DIANA",
                ),
            ),  # upper clip is updated during training
            (
                {"types": ("Conv2d")},
                (
                    "per-array",
                    {"bitwidth": 8, "signed": True},
                    qhparamsinitstrategy,
                    "DIANA",
                ),
            ),  # can use per-outchannel_weights here
            (
                {"types": ("Linear")},
                (
                    "per-array",
                    {"bitwidth": 8, "signed": True},
                    qhparamsinitstrategy,
                    "DIANA",
                ),
            ),
        )
        # Edit the following line if you want to use the quant stepper
        analogcoredescriptionspec = ("per-array", "ternary", "meanstd")
        # analogcoredescriptionspec =  ('per-array', {'bitwidth': 8 , 'signed': True} , 'meanstd' )
        # analogcoredescriptionspec =  ('per-array', {'bitwidth': 3 , 'signed': True} , 'meanstd' )

        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = (
            "per-array",
            {"bitwidth": 8, "signed": True},
            qhparamsinitstrategy,
            "DIANA",
        )
        graph = qg.fx.quantlib_symbolic_trace(root=model)  # graph module

        converter = F2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            analogcoredescriptionspec,
            modules_descriptors=modules_descriptors,
        )

        converted_graph = converter(graph)

        return converted_graph

    def set_quantized(self, activations):
        """
        dataset_item (torch.Tensor):  tensor with batch size >= 1
        """
        if activations:
            self.start_observing()
        else:
            self.start_observing(not_modules=_QActivation)

        for data in self.representative_dataset():
            _ = self.gmodule(data)

        if activations:
            self.stop_observing()
        else:
            self.stop_observing(not_modules=_QActivation)

        # Before converting scales to power-of-two, we must also freeze the clipping bounds of ReLU, otherwise
        # the DIANAReLU scale may be altered during the next forward pass
        self.freeze_clipping_bound()

        # Convert scales to power-of-two
        self.convert_scales_to_pot()

        # Set input scales of element-wise-add operations to the same values
        self.align_ews_input_scales()

    def map_to_hw(self, custom_editors: List[Editor] = []):
        """
        custom_editors: optional additional editors
        """
        converter = HWMappingConverter(custom_editor=custom_editors)

        dataset_item = next(self.representative_dataset())
        with torch.no_grad():
            self.gmodule = converter(
                self.gmodule,
                {
                    "x": {
                        "shape": dataset_item.shape,
                        "scale": self.dataset_scale,
                    }
                },
                input=dataset_item,
            )

    def integrize_layers(self):
        """
        """
        converter = LayerIntegrizationConverter()

        dataset_item = next(self.representative_dataset())
        with torch.no_grad():
            self.gmodule = converter(
                self.gmodule,
                {
                    "x": {
                        "shape": dataset_item.shape,
                        "scale": self.dataset_scale,
                    }
                },
            )
        self._integrized = True

    def export_model(self, export_folder: str):
        """
        export_folder: folder name to export. File name is derived from the model name.
        """
        # x is an integrised tensor input is needed for validation in dory graph
        if not self._integrized:
            raise NotImplementedError

        exporter = DianaExporter()
        from pathlib import Path

        dataset_item = next(self.representative_dataset())

        # ensure the item has batchsize 1
        if dataset_item.size(0) > 1:
            dataset_item = dataset_item[0].unsqueeze(0)

        x = (dataset_item / self.dataset_scale).floor()  # integrize

        exporter.export(
            network=self.gmodule, input_shape=x.shape, path=export_folder
        )
        exporter.dump_features(network=self.gmodule, x=x, path=export_folder)

    @classmethod
    def return_DFS(cls, node: fx.node.Node, depth: int):
        ls: List[fx.node.Nodes] = []
        n = node
        for i in range(depth):
            users = [u for u in n.users]
            if len(users) == 0:
                return ls
            ls.append(users[0])
            n = users[0]
        return ls

    @classmethod
    def remove_dict_prefix(cls, old_state_dict, start=8):
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            name = k[start:]  # remove `gmodule.`
            new_state_dict[name] = v

        return new_state_dict

    @classmethod
    def remove_prefixes(cls, old_state_dict):
        words = ["module", "student"]
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():

            for word in words:
                start = k.find(word)
                if start != -1:
                    start = start + len(word) + 1
                    break
            if start != -1:
                name = k[start:]  # remove `module.`
                new_state_dict[name] = v

        return new_state_dict
