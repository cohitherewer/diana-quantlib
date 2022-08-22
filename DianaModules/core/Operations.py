
from abc import abstractmethod
import enum
from turtle import forward
import torch 
from torch import Tensor, nn
from typing import Union , Tuple


from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import PACTReLU
from quantlib.algorithms.qbase.observer.observers import TensorObserver
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import resolve_qrangespec



from quantlib.algorithms.qmodules.qmodules import  QIdentity


from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear 

import math

from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from DianaModules.utils._FakeQuantizer import _FakeDQuantiser, _FakeAQuantiser
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.algorithms.qbase import get_zero_scale, get_scale

class DianaBaseOperation:   
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): 
        pass
    
    def redefine_qhparams(self : _QModule, qrangespec:               QRangeSpecType):  
        assert(issubclass(type(self), _QModule))
        device = self.zero.device 
        self._qrange = resolve_qrangespec(qrangespec)
        zero, n_levels, step, scale = create_qhparams(self._qrange)
        self.zero =  torch.tile(zero,     self.zero.shape).to(device)
        self.n_levels=  torch.tile(n_levels,     self.n_levels.shape).to(device)
        self.step=  torch.tile(step,    self.step.shape).to(device)
        self.scale =  torch.tile(scale,   self.scale.shape).to(device)
        if self._pin_offset:
            scale = get_scale(self.min_float, self.max_float, self.zero, self.n_levels, self.step)
            self.scale.data.copy_(scale.to(device=self.scale.device))
        else:
            zero, scale = get_zero_scale(self.min_float, self.max_float, self.n_levels, self.step)
            self.zero.data.copy_(zero.to(device=self.scale.device))
            self.scale.data.copy_(scale.to(device=self.scale.device))
        self._set_clipping_bounds()
    
    def define_bitwidth_clipping(self): 
        self.bw_clip_lo = torch.exp2(torch.round(torch.log2((torch.abs(self.clip_lo)/ (self.scale * self.step)))) )#quantized clip lo and high
        self.bw_clip_hi =torch.exp2(torch.round(torch.log2((torch.abs(self.clip_hi)/ (self.scale * self.step)))) )-1 
        if len(self.clip_lo.shape ) > 3: 
                for c in range(self.clip_lo.size(0)) : 
                    if self.clip_lo[c][0][0][0] < 0 : 
                        self.bw_clip_lo[c][0][0][0] =  -self.bw_clip_lo[c][0][0][0] 
                      
        else: 
            if self.clip_lo < 0: 
                self.bw_clip_lo = -self.bw_clip_lo 
        if self.clip_lo < 0: 
            self.bw_clip_lo = -self.bw_clip_lo + 1
        
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
               
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 
    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        return self._qop(x, self.bw_clip_lo,self.bw_clip_hi, self.step, self.scale)
    
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})                     
        self.define_bitwidth_clipping()

    def forward(self, x: torch.Tensor) : 
        x = x.to(self.weight.device)
        return super().forward(x) 
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
    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        
        return self._qop(x, self.bw_clip_lo,self.bw_clip_hi, self.step, self.scale)
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
        self.define_bitwidth_clipping()
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
    def __init__(self, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType, inplace: bool = False , is_analog : bool = False):
        self.is_analog = is_analog
        super().__init__(qrangespec, qgranularityspec, qhparamsinitstrategyspec, inplace)
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if  self.is_analog:
                self.redefine_qhparams({'bitwidth' : 24, 'signed': False})
            else: 
                self.redefine_qhparams({'bitwidth' : 7, 'signed': False})

            #  clip here and freeze 
            self.freeze() 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 
        self.define_bitwidth_clipping()
    

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
                    self.scaled_mapped=False #TODO change this later
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
                         bias=bias) 

    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        if self.is_analog:
             self._qop = _FakeAQuantiser.apply
        else: 
            self._qop = _FakeDQuantiser.apply 
 
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        

        return self._qop(x, self.bw_clip_lo, self.bw_clip_hi, self.step, self.scale) 
   
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if (self.is_analog): 
                self.redefine_qhparams('ternary')
                self.bw_clip_lo = torch.tile(torch.Tensor([-1]) , self.clip_lo.shape).to(self.clip_lo.device)     
                self.bw_clip_hi = torch.tile(torch.Tensor([1]) , self.clip_hi.shape).to(self.clip_lo.device)  
                return 
            else: 
                self.redefine_qhparams({'bitwidth' : 8 , 'signed': True})   

        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})   
        self.define_bitwidth_clipping()
   




# Analog Core Conv Operation: DAC , noise ,  split channels conv ,  quantize , then accumulate 
class AnalogConv2d(DIANAConv2d): 
    def __init__(self, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType, in_channels: int, out_channels: int, kernel_size: int, stride: Tuple[int, ...] = 1, padding: str = 0, dilation: Tuple[int, ...] = 1, array_size : int = 1152):
        self.max_channels = math.floor(array_size/kernel_size *kernel_size)
        assert self.max_channels >= 1 , f"Array size must be at least kernel_size * kernel_size: {kernel_size**2}"
        self.gain = 1 # Analog domain gain from Vgs 
        self.out_H = None # saving to not keep recomputing in forward pass 
        self.out_W = None # saving to not keep recomputing in forward pass 
        super().__init__(qrangespec, qgranularityspec, qhparamsinitstrategyspec, in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)

    def forward(self, x : torch.Tensor) : # returns a five dimensionsal tensor 
        x.to(self.weight.device) 
        group_count = 1 if self.max_channels >= x.size(1) else math.ceil(x.size(1)/self.max_channels) 
        counter = x.size(1) 
        tile_size = x.size(1) if group_count > 1 else self.max_channels
        padded_out = None
        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight
        feature_map_H = x.size(2)  
        feature_map_W = x.size(3) 
        kernel_count = self.out_channels
        padding = self.padding
        stride = self.stride
        for i in range(group_count): # chance for parallelization across distributed loads 
            group_passed = x[: , x.size(1)*(i):tile_size, : , : ] 
            
            ##### TWO LINE CONVOLUTION #####
            
            feature_map_batch_size = group_passed.size(0) 
            
            flattened_feature_map = torch.nn.functional.unfold(group_passed ,kernel_size =self.kernel_size,stride=self.stride , padding=self.padding ).unsqueeze(1) # 
            flattened_feature_map = flattened_feature_map.expand(feature_map_batch_size , kernel_count , flattened_feature_map.size(2) , flattened_feature_map.size(3) )
            print(flattened_feature_map.shape)
            flattened_kernel = weight.view(weight.size(0), -1).unsqueeze(0).expand(feature_map_batch_size , kernel_count , -1).unsqueeze(3).expand(feature_map_batch_size,kernel_count, flattened_feature_map.size(2), flattened_feature_map.size(3))

            flattened_out = flattened_kernel * flattened_feature_map
            out_H = int((feature_map_H-self.kernel_size+2* padding)/stride+1 ) if self.out_H is None else self.out_H
            out_W = int((feature_map_W-self.kernel_size+2* padding)/stride+1 ) if self.out_W is None else self.out_W 

            negative_mask = flattened_out <0


            negative_clip = -10 
            positive_clip = 10

            negative_out = torch.max(torch.sum(flattened_out*negative_mask , 2)  , negative_clip) 
            positive_out = torch.min(torch.sum(flattened_out*~negative_mask , 2), positive_clip) 

            conv_out = (negative_out + positive_out).reshape(feature_map_batch_size ,kernel_count, out_H ,out_W )


            ##### TWO LINE CONVOLUTION END #####
            #conv_out = super().forward(group_passed) 
            if padded_out is None:            
                padded_out = torch.zeros(group_count , x.size(0) , self.out_channels , conv_out.size(2)  ,conv_out.size(3)  ).to(self.weight.device) 
                self.out_W = conv_out.size(3) 
                self.out_H = conv_out.size(2) 

            padded_out[i,:, :conv_out.size(1), : , : ] = conv_out
            counter -= tile_size 
            if counter < self.max_channels: 
                tile_size = counter # remaining 
        return padded_out
        
 
class AnalogOutIdentity(DIANAIdentity):  ## keeping an observer for  each of the groups except the last one ( because it might be padded with zeros and might affect the statistics). In the end fuse the observer statistics 
    def __init__(self, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        super().__init__(qrangespec, qgranularityspec, qhparamsinitstrategyspec, IdentityType.AIMC_OUT)
    def forward(self, x : torch.Tensor):
        pass 
        
#(Partial sum ) 
class Accumulator(nn.Module): # given a five dimensional tensor return a reduced accumulated 4d tensor 
    def __init__(self) -> None:
        super().__init__()
    def forward(self , x: torch.Tensor) : 
        return torch.sum(x , 0) 

class GaussianNoise(nn.Module) : 
    def __init__(self) -> None:
        super().__init__()
    def forward(self , x: torch.Tensor) : 
        pass 
