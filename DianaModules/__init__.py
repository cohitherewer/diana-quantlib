from quantlib.algorithms.qalgorithms import  register
from  DianaModules.Digital.DIlayers import DIANAConv2d ,DIANAIdentity, DIANALinear , DIANAReLU
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import ModuleMapping
NNMODULE_TO_DIANA = ModuleMapping ([
    (nn.Identity,  DIANAIdentity),
    (nn.ReLU,      DIANAReLU),
   (nn.Linear , DIANALinear ), 

    (nn.Conv2d , DIANAConv2d) ])
    
register['DIANA'] = NNMODULE_TO_DIANA
