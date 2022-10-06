from quantlib.algorithms.qalgorithms import  register
from  DianaModules.core.Operations import DIANAConv2d ,DIANAIdentity, DIANALinear, DIANAReLU 
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import ModuleMapping
NNMODULE_TO_DIANA = ModuleMapping ([
    (nn.Identity,  DIANAIdentity),
    (nn.ReLU , DIANAReLU),
   (nn.Linear , DIANALinear ), 
    (nn.Conv2d , DIANAConv2d) 
    ])
    
register['DIANA'] = NNMODULE_TO_DIANA


from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.serialization.Loader import ModulesLoader


def from_torch_model(model, descriptors_file=None):
    module_descriptions = None
    if descriptors_file is not None:
        loader = ModulesLoader()
        module_descriptions = loader.load(module_descriptions_pth) 

    fake_quantized_model = DianaModule.from_trainedfp_model(model=model , modules_descriptors=module_descriptions)
    return DianaModule(fake_quantized_model)