from typing import List
import yaml
from torch import nn
import torch.fx as fx
from DianaModules.utils.converters.float2fake import F2FConverter
from quantlib.algorithms.qbase import qgranularity, qhparamsinitstrategy
from quantlib.algorithms.qbase.qgranularity.qgranularity import QGranularity
from quantlib.algorithms.qbase.qhparamsinitstrategy.qhparamsinitstrategy import (
    ConstInitStrategy,
    MinMaxInitStrategy,
)
from .Descriptor import ModuleDescriptor
from DianaModules.core.Operations import (
    DianaBaseOperation,
    DIANAConv2d,
    DIANALinear,
    AnalogConv2d,
)


class ModulesSerializer:
    def __init__(self, fq_model: fx.graph_module.GraphModule) -> None:

        # Create list of descriptors
        self.descriptors: List[ModuleDescriptor] = []
        # iterate thorugh nodes in graph and populate descripts list
        for node in fq_model.graph.nodes:
            try:
                module = fq_model.get_submodule(node.target)
                if isinstance(module, DianaBaseOperation):
                    if isinstance(
                        module, (DIANAConv2d, DIANALinear)
                    ):  # for now depth-wise sep not supported
                        core_choice = (
                            "DIGITAL"
                            if type(module) != AnalogConv2d
                            else "ANALOG"
                        )  # TODO When depthwise seperable convolutions are added , implement the feature here
                        if type(module._qinitstrategy) == ConstInitStrategy:
                            qhparamsinitstrategyspec = {
                                "CONST": module.default_kwargs
                            }
                        elif type(module._qinitstrategy) == MinMaxInitStrategy:
                            qhparamsinitstrategyspec = "MINMAX"
                        else:
                            qhparamsinitstrategyspec = "MEANSTD"
                        if module._qgranularity == QGranularity((0,)):
                            qgranspec = "PER-OUTCHANNEL_WEIGHTS"
                        else:
                            qgranspec = "PER-ARRAY"
                        descriptor = ModuleDescriptor(
                            node.target,
                            core_choice,
                            qhparamsinitstrategyspec,
                            qgranspec,
                            module.weight.size(),
                        )
                        self.descriptors.append(descriptor)
            except:
                pass
        pass

    def dump(self, path: str):
        # dump all
        with open(path, "w") as f:
            f.write(yaml.dump_all(self.descriptors))
        pass
