from quantlib.algorithms.qalgorithms import  register
from  DianaModules.core.operations import DIANAConv2d ,DIANAIdentity, DIANALinear 
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import ModuleMapping
NNMODULE_TO_DIANA = ModuleMapping ([
    (nn.Identity,  DIANAIdentity),
   (nn.Linear , DIANALinear ), 
    (nn.Conv2d , DIANAConv2d) ])
    
register['DIANA'] = NNMODULE_TO_DIANA
