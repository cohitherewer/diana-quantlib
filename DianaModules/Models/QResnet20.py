import torch 
from torch import nn 
from CompositeBlocks.Resblock import QResblock
from DianaModules.utils.DianaModule import DianaModule
from quantlib.algorithms.qbase import QRangeSpecType , QGranularitySpecType, QHParamsInitStrategySpecType


class QResnet20(DianaModule): 
    def __init__(qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType): 
                 pass 
                 
    
    pass 


