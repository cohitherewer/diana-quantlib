from quantlib.algorithms.qalgorithms import  register
from DianaModules.Digital.DIlayers import DIANAFC, DIANABatchNorm2d, DIANAIdentity, DIANAPool2d, DIANAReLU
from DianaModules.AnIA.ANAlayers import DIANAConv2d
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import ModuleMapping
NNMODULE_TO_DIANA = ModuleMapping ([
    (nn.Identity,  DIANAIdentity),
    (nn.ReLU,      DIANAReLU),
    (nn.Linear , DIANAFC), 
    (nn.BatchNorm2d, DIANABatchNorm2d) , 
    #(nn.AvgPool2d , DIANAPool2d) not supported by quantlib will have to add
    (nn.Conv2d , DIANAConv2d) ])
    
register['DIANA'] = NNMODULE_TO_DIANA
