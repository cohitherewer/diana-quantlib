
from abc import abstractmethod
import enum
from numpy import require

import torch 
from torch import Tensor, nn
from typing import Union , Tuple


from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import PACTReLU
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import resolve_qrangespec



from quantlib.algorithms.qmodules.qmodules import  QIdentity
from quantlib.algorithms.qmodules.qmodules.qactivations import QReLU

from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear 

import math

from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from DianaModules.utils._FakeQuantizer import _FakeDQuantiser, _FakeAQuantiser
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule

class DianaBaseOperation:   
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): 
        pass
    
    def redefine_qhparams(self : _QModule, qrangespec:               QRangeSpecType):  
        assert(issubclass(type(self), _QModule))
        self._qrange = resolve_qrangespec(qrangespec)
        zero, n_levels, step, scale = create_qhparams(self._qrange)
        self.zero =  torch.tile(zero,     self._observer.broadcasting_shape)
        self.n_levels=  torch.tile(n_levels,     self._observer.broadcasting_shape)
        self.step=  torch.tile(step,    self._observer.broadcasting_shape)
        self.scale =  torch.tile(scale,   self._observer.broadcasting_shape)
        self._init_qhparams() # reinitialize scale and zero 
    def get_bitwidth(self): 
        pass

class IdentityType(enum.Enum):  
    default = enum.auto() # 8 Bits 
    AIMC_IN= enum.auto() # 7 bits
    AIMC_OUT= enum.auto() # 6 bits 
    DIGITAL_IN = enum.auto() # 8 bits (digital core)
    DIGITAL_OUT = enum.auto() # 8 bits 

class DIANALinear(QLinear , DianaBaseOperation): 
    def __init__(self, qrangespec : QRangeSpecType,
                 qgranularityspec : QGranularitySpecType,
                 qhparamsinitstrategyspec : QHParamsInitStrategySpecType,in_features : int, out_features: int, bias=False): 
                 super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features, bias=bias) 
                 self.relu_output = False 
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 
    def set_relu_on (self) :
        self.relu_output = True 
    def is_relu_on(self) : 
        return self.relu_output
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        bw_clip_lo = 2**round(math.log2((abs(self.clip_lo)/ (self.scale * self.step)).item())) #quantized clip lo and high
        bw_clip_hi =2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) -1 
        if self.clip_lo < 0: 
            bw_clip_lo = -bw_clip_lo + 1
        return self._qop(x, bw_clip_lo,bw_clip_hi, self.step, self.scale)
    
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})              
# Activations
class DIANAIdentity(QIdentity , DianaBaseOperation): # general purpose identity for harmoniser adds ( Quant operation )
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,
                 QuantizerType : IdentityType = IdentityType.default):

        super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
        self._type =  QuantizerType
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        bw_clip_lo = 2**round(math.log2((abs(self.clip_lo)/ (self.scale * self.step)).item())) #quantized clip lo and high
        bw_clip_hi =2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) -1 
        if self.clip_lo < 0: 
            bw_clip_lo = -bw_clip_lo +1
        return self._qop(x, bw_clip_lo,bw_clip_hi, self.step, self.scale)
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if self._type == IdentityType.default: 
                self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
            elif self._type == IdentityType.AIMC_IN: 
                self.redefine_qhparams({'bitwidth' : 7, 'signed': True}) 
            elif self._type == IdentityType.AIMC_OUT: 
                self.redefine_qhparams({'bitwidth' : 6, 'signed': True}) 
            else:
                self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 
    def get_bitwidth(self):
        if self._type == IdentityType.default: 
            return 8 
        elif self._type == IdentityType.AIMC_IN: 
            return 7 
        elif self._type == IdentityType.AIMC_OUT: 
            return 6 
        else:
            return 8 

class DIANAReLU( PACTReLU  , DianaBaseOperation): 
    def stop_observing(self):
        super().stop_observing() 
        self.bw_clip_hi = 2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) - 1
        self.freeze()
        # edit the scale

    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):

        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 7, 'signed': True})

            #  clip here and freeze 
            self.freeze() 
            

            
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 



# How I have it right now it will be a convolution in the digital core if it's not followed by a batch norm otherwise it's an analog core
class DIANAConv2d(QConv2d , DianaBaseOperation):
    
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
                 groups:                   int = 1 , 
                 bias : bool = False 
                 ):
                    self.is_analog = not bias # In linearopbn cannonocalisation the bias is set to none if bn follows conv 
                    self.relu_output = False 
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
                         bias=bias) # you can remove all this bias stuff 

          
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        if self.is_analog:
             self._qop = _FakeAQuantiser.apply
        else: 
            self._qop = _FakeDQuantiser.apply 
 
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale) 
    def set_relu_out (self) :
        self.relu_output = True 
    def is_relu_out(self) : 
        return self.relu_output
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if (self.is_analog): 
                self.redefine_qhparams('ternary') 
            else: 
                self.redefine_qhparams({'bitwidth' : 8 , 'signed': True})   

        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})   




