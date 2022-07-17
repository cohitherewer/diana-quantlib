from abc import abstractmethod
from multiprocessing import pool
import torch 
from torch import nn
from typing import Union , Tuple


from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import PACTReLU

from DianaModules.utils.DianaModule import DianaBaseOperation, DianaModule

from quantlib.algorithms.qmodules.qmodules import  QIdentity

from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear 

import math

from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from DianaModules.utils._FakeQuantizer import _FakeDQuantiser



# later define hardware specific quantisation 


class DQLinear(QLinear , DianaBaseOperation): 
    def __init__(self, qrangespec : QRangeSpecType,
                 qgranularityspec : QGranularitySpecType,
                 qhparamsinitstrategyspec : QHParamsInitStrategySpecType,in_features : int, out_features: int, bias=False): 
                 super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features, bias=False) 
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 

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

class DQConv2d(QConv2d , DianaBaseOperation) : 
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
                 groups:                   int = 1
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
        self._qop = _FakeDQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)
    
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})   
# Activations
class DIANAIdentity(QIdentity , DianaBaseOperation): 
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
    
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
            self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 

class DIANAReLU( PACTReLU  , DianaBaseOperation): 
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 7, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 
    
# Layers # TODO All layers should also quantize the output 
class DIANABatchNorm2d(DianaModule): # output not quantised 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:             torch.Tensor): #channels * batch_size  
        super().__init__() 
        self.qscale = DIANAIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( (torch.rand(in_channels) -0.5)*2))
        self.register_parameter(name='weights', param = torch.nn.parameter.Parameter( (torch.rand(in_channels)-0.5 )* 2))
        self.qidentity = DIANAIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 

    @property 
    def qweight(self): 
        return self.qscale(self.weights)
    @property 
    def qbias(self): 
        return self.qidentity(self.biases)
    def get_weight_scale(self): 
        return self.qscale.scale
    def get_bias_scale(self): 
        return self.qidentity.scale
    @classmethod
    def from_fp_module(cls, bn_layer : nn.BatchNorm2d, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        return DIANABatchNorm2d(qrangespec, qgranularityspec, qhparamsinitstrategyspec, bn_layer.num_features)
    def forward(self, input):
        ## Broadcasting to get weights/biases in the correct format 
        broadcasted_biases = self.qidentity(self.biases).unsqueeze(1).unsqueeze(1)
        broadcasted_weights = self.qscale(self.weights).unsqueeze(1).unsqueeze(1)
       # if (len(input.size())) > 1: 
            #broadcasted_weights = broadcasted_weights.unsqueeze(1)
            #broadcasted_biases= broadcasted_biases.unsqueeze(1)
            #if(len(input.size())) > 2: 
                #broadcasted_weights = broadcasted_weights.unsqueeze(1)
                #broadcasted_biases= broadcasted_biases.unsqueeze(1)
        
        broadcasted_weights = broadcasted_weights.expand(input.size())
        broadcasted_biases = broadcasted_biases.expand(input.size())

        return input * broadcasted_weights+ broadcasted_biases

# pooling layer use regular pool and then have it go through Qidentity 
class DIANAPool2d(DianaModule): 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,kernel_size, stride=None, padding=0) : 
        super().__init__() 
        self.qidentity = DIANAIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size,stride = stride, padding=padding) # TODO: Implement the custom avgpool 
    @classmethod
    def from_fp_module(cls, pool_layer : nn.AvgPool2d, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        return DIANAFC(qrangespec, qgranularityspec, qhparamsinitstrategyspec, pool_layer.kernel_size, stride =pool_layer.stride , padding = pool_layer.padding)
    def forward(self, input) : 
        out = self.avgpool(input)
        out = self.qidentity(out) 
        return out 

class DIANAFC(DianaModule): # output quantized #TODO do I Quantized bias ? 
    def __init__(self,qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType , in_features: int , out_features: int): 
        super().__init__()
        print(type(qrangespec), type(qgranularityspec), type(qhparamsinitstrategyspec), type(in_features), type(out_features))
        self.qlinear = DQLinear(qrangespec=qrangespec,qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features)
        self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_features) -0.5)) # think of another way to initialise the biases 
        self.qidentity = DIANAIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
        self.qout = DIANAIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
    @classmethod
    def from_fp_module(cls, linear_layer : nn.Linear, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        return DIANAFC(qrangespec=qrangespec, qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_features=linear_layer.in_features, out_features=linear_layer.out_features)

    def forward(self, input : torch.Tensor): 
        broadcasted_input = input.flatten(start_dim=1) 
        broadcasted_biases = self.qidentity(self.biases).expand(input.size(0),)# broadcasted for batches 
        return self.qout(self.qlinear(broadcasted_input)  + broadcasted_biases) # make sure input sizes match 

class DIANAConv2d_d(DianaModule): # default bias false , implemented withoutput quantizer. 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  int = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 bias : bool = False ):
        super().__init__() 
        self.qin = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
        self.qconv = DQConv2d(qrangespec=qrangespec,qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec,kernel_size=kernel_size,stride=stride, padding=padding,in_channels=in_channels ,out_channels=out_channels, dilation=dilation, groups=groups, bias = bias) 
        self.qout = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec)
        self.bias_enabled = bias 
        if self.bias_enabled: 
            self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_channels) -0.5)) # think of another way to initialise the biases 
            self.qbiasin = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
 
    def forward(self, input): 
        broadcasted_bias = 0 
        if self.bias_enabled: # still need to initialize the bias correctly. Right now it's not initialized correctly 
            broadcasted_bias = self.qbiasin(self.biases)
        output = self.qout(self.qconv(self.qin(input)) + broadcasted_bias)
        return output

#In training scales are already matched no custom res_add is needed, but when doing inference scales need to be accounted for 



