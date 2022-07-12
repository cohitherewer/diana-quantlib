import torch 
from torch import nn 
from CompositeBlocks.Resblock import QResblock
from DianaModules.utils.DianaModule import DianaModule
from quantlib.algorithms.qbase import QRangeSpecType , QGranularitySpecType, QHParamsInitStrategySpecType


class QResnet20(nn.Module, DianaModule): 
    def __init__(qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType): 
                 pass 
    def start_observing(self):
            return super().start_observing()
    def stop_observing(self):
            return super().stop_observing()
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
            return super().map_scales(new_bitwidth, signed, HW_Behaviour)
    def clip_scales(self):
            return super().clip_scales() 
    
    pass 


