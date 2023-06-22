from quantlib.algorithms.qalgorithms import register
from dianaquantlib.core.Operations import (
    DIANAConv2d,
    DIANAIdentity,
    DIANALinear,
    DIANAReLU,
)
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import (
    ModuleMapping,
)

NNMODULE_TO_DIANA = ModuleMapping(
    [
        (nn.Identity, DIANAIdentity),
        (nn.ReLU, DIANAReLU),
        (nn.Linear, DIANALinear),
        (nn.Conv2d, DIANAConv2d),
    ]
)

register["DIANA"] = NNMODULE_TO_DIANA


from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Loader import ModulesLoader


def from_torch_model(model, descriptors_file=None):
    module_descriptions = None
    if descriptors_file is not None:
        loader = ModulesLoader()
        module_descriptions = loader.load(descriptors_file)

    fake_quantized_model = DianaModule.from_trainedfp_model(
        model=model, modules_descriptors=module_descriptions
    )
    return DianaModule(fake_quantized_model)
