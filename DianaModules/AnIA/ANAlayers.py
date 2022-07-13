# To be done later is automatically convert neural network architecture (using quantlib editing) from true to fake quantised (still using float respresentation to represent quantised values 
# )to ultimately true quantised networks. , Priority: low 
# Add tools to be able to implemnt autograd quants functions from algorithms  
# (can be implementeda as sequential modules) Added noise based on model from aihwkit needed, Priority: high 
import torch
from torch import nn

from typing import Tuple
from DianaModules.utils.DianaModule import DianaModule, DianaBaseOperation


from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from  quantlib.algorithms.qmodules.qmodules import QConv2d, QIdentity
from utils._FakeQuantizer import _FakeAQuantiser
# qbitwidth inputs: 7bits, outputs:   6b 
# weights: in: 32bits , out: 2bits 
# 
# no bias 
# quantisation: optional between per-layer (per-output array) or per array , in images per channel could be useful 

# HARDWARE SPECS 
#output quantization
IDENTITY_RANGE_SPEC =  {'bitwidth': 6, 'signed': True}
IDENTITY_GRANULARITY_SPEC = 'per-array' # cannot be changed (per layer)
IDENTITY_INITSTRAT_SPEC = 'meanstd' # clipping at mean + 3* std and mean -3std could also be const or minmax 
# Weights quantization
CONV_RANGE_SPEC = 'ternary'
CONV_GRANULARITY_SPEC ='per-array' # can also be 'per-outchannel_weights' 
CONV_INITSTRAT_SPEC = 'meanstd'


class AQConv2D(QConv2d, DianaBaseOperation): 
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  int = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 ):
                 super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias=False)
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeAQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)  
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: #defaults hardware parameters ternary behaviour
            # mapping qh parameters 
            self.redefine_qhparams( 'ternary')
        else: 
            self.redefine_qhparams( {'bitwidth':new_bitwidth , 'signed': signed})

class AQIdentity(QIdentity , DianaBaseOperation): 
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeAQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)           
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: #defaults hardware parameters ternary behaviour
            self.redefine_qhparams({'bitwidth' : 6, 'signed': True}) 
        else : 
            self.redefine_qhparams( {'bitwidth':new_bitwidth , 'signed': signed})
                         
class AnIAConvLayer(nn.Module, DianaModule): # output is quantized 
    def __init__(self,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  str = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,

                 
                 ): 
        self.qconv = AQConv2D(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=False, qrangespec=CONV_RANGE_SPEC,
                          qgranularityspec=CONV_GRANULARITY_SPEC,
                          qhparamsinitstrategyspec=CONV_INITSTRAT_SPEC) 
        self.qidentity = AQIdentity(IDENTITY_RANGE_SPEC, IDENTITY_GRANULARITY_SPEC, IDENTITY_INITSTRAT_SPEC)
        
    def start_observing(self): 
        self.qconv.start_observing() 
        self.qidentity.start_observing()
         #when stopping observations scale can be checked, modified and rounder to closest power of 2 #DONE with map scales   
    def stop_observing(self): #initialises the quantization hyperparameters and model becomes quantized
        self.qconv.stop_observing() 
        self.qidentity.stop_observing()
    def forward(self, input) : 
        return self.qidentity(self.qconv(input) ) 
  

