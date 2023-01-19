# Abstract class which all diana specific module have to inhereit from
from abc import abstractmethod
from cProfile import run
from collections import OrderedDict
from math import log2, floor
from pathlib import Path
from random import randint
import sched
from typing import Any, Dict, List, Union

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
        graph_module: Union[nn.Module, fx.graph_module.GraphModule] = None,
        input_shape=None,
    ):
        super().__init__()
        self.gmodule = graph_module
        self._integrized = False
        self._input_shape = input_shape

    def start_observing(
        self, is_modules=(_QModule, HarmonisedAdd), not_modules=type(None)
    ):  # before starting training with FP
        for _, module in self.gmodule.named_modules():
            if isinstance(module, is_modules) and not isinstance(
                module, not_modules
            ):
                module.start_observing()

    def stop_observing(
        self, is_modules=(_QModule, HarmonisedAdd), not_modules=type(None)
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

    @classmethod
    def from_trainedfp_model(
        cls, model: nn.Module, modules_descriptors=None
    ):  # returns fake quantized model from_ floating point quantised model
        modulewisedescriptionspec = (  # change const later
            (
                {"types": ("Identity")},
                (
                    "per-array",
                    {"bitwidth": 8, "signed": True},
                    "meanstd",
                    "DIANA",
                ),
            ),
            (
                {"types": ("ReLU")},
                (
                    "per-array",
                    {"bitwidth": 7, "signed": False},
                    "meanstd",
                    "DIANA",
                ),
            ),  # upper clip is updated during training
            (
                {"types": ("Conv2d")},
                (
                    "per-outchannel_weights",
                    {"bitwidth": 8, "signed": True},
                    "meanstd",
                    "DIANA",
                ),
            ),  # can use per-outchannel_weights here
            (
                {"types": ("Linear")},
                (
                    "per-outchannel_weights",
                    {"bitwidth": 8, "signed": True},
                    "meanstd",
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
            "meanstd",
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

    def set_quantized(self, activations, dataset=None, batch_size=128):
        if activations:
            self.start_observing()
        else:
            self.start_observing(not_modules=_QActivation)

        if dataset is not None:
            data_loader = ut.DataLoader(
                dataset, batch_size=batch_size, pin_memory=True, shuffle=True
            )
            for idx, data in enumerate(data_loader):
                x, y = data[0], data[1]
                _ = self.gmodule(x)
        else:
            x = torch.randn(*self._input_shape).unsqueeze(0)
            _ = self.gmodule(x)

        if activations:
            self.stop_observing()
        else:
            self.stop_observing(not_modules=_QActivation)

        for _, module in self.gmodule.named_modules():
            if (
                type(module) == DIANAConv2d
                or isinstance(module, DIANALinear)
                or (
                    isinstance(module, _QActivation)
                    and not isinstance(module, AnalogOutIdentity)
                )
            ):
                module.scale = torch.Tensor(
                    torch.exp2(torch.round(torch.log2(module.scale)))
                )

    def map_to_hw(
        self,
        dataset_scale,
        dataset=None,
        custom_editors: List[Editor] = [],
    ):
        # free relus upper bound
        for _, mod in self.named_modules():
            if isinstance(mod, DIANAReLU):
                mod.freeze()
        converter = HWMappingConverter(custom_editor=custom_editors)

        # does it matter which data is forwarded?
        if dataset is None:
            x = torch.randn(*self._input_shape)
        else:
            x = dataset.__getitem__(0)[0]

        if len(x.shape) == 3:  # TODO: always unsqueeze?
            x = x.unsqueeze(0)

        with torch.no_grad():
            self.gmodule = converter(
                self.gmodule,
                {
                    "x": {
                        "shape": x.shape,
                        "scale": dataset_scale,
                    }
                },
                input=x,
            )

    def integrize_layers(
        self,
        dataset_scale,
        dataset=None,
    ):
        converter = LayerIntegrizationConverter()

        # does it matter which data is forwarded?
        if dataset is None:
            x = torch.randn(*self._input_shape)
        else:
            x = dataset.__getitem__(0)[0]

        if len(x.shape) == 3:  # TODO: always unsqueeze?
            x = x.unsqueeze(0)

        x.to("cpu")
        with torch.no_grad():
            self.gmodule = converter(
                self.gmodule,
                {
                    "x": {
                        "shape": x.shape,
                        "scale": dataset_scale,
                    }
                },
            )
        self._integrized = True

    def export_model(
        self,
        export_folder: str,
        dataset_scale,
        dataset=None,
    ):
        # x is an integrised tensor input is needed for validation in dory graph
        if not self._integrized:
            raise NotImplementedError

        exporter = DianaExporter()
        from pathlib import Path

        # does it matter which data is forwarded?
        if dataset is None:
            x = torch.randn(*self._input_shape)
        else:
            x = dataset.__getitem__(0)[0]

        if len(x.shape) == 3:  # TODO: always unsqueeze?
            x = x.unsqueeze(0)

        x = (x / dataset_scale).floor()  # integrize

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
            name = k[start:]  # remove `module.`
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
