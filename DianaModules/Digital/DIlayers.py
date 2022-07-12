from abc import abstractmethod
import torch 
from torch import nn
from typing import Union , Tuple
from DianaModules.utils.DianaModule import DianaModule

from quantlib.algorithms.qmodules.qmodules import  QIdentity
from quantlib.algorithms.qmodules.qmodules.qactivations import QReLU, QReLU6
from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear 


from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from utils._FakeQuantizer import _FakeDQuantiser
# later define hardware specific quantisation 

#scale is the value you divide the flaoting value with to get the quantised range to get the original value 



# NEED to test with nn.linear if we input 4D tensor and have input & output features equal what happens 

# add quantized identity 
class DQIdentity(QIdentity): 
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
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)  




class DQScaleBias(nn.Module, DianaModule): # output not quantised 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:             torch.Tensor): #channels * batch_size  
        super().__init__() 
        self.qscale = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(in_channels) -0.5))
        self.register_parameter(name='weights', param = torch.nn.parameter.Parameter( torch.randn(in_channels)))
        self.qidentity = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 


    def start_observing(self): 
        self.qscale.start_observing()
        self.qidentity.start_observing() 
  
      #when stopping observations scale can be checked, modified and rounder to closest power of 2 #TODO
      #this is implied   i_scale*w_scale*b_scale > 1 if above criteria is fulfilled 
    def stop_observing(self):
        self.qscale.stop_observing()
        self.qidentity.stop_observing() 
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if not self.qscale._is_quantised or not self.qidentity._is_quantised :  # should only be called after stop_observing() 
            # not quantized 
            return 
        if HW_Behaviour: #defaults hardware parameters ternary behaviour
            # mapping qh parameters 
         
            DianaModule.redefine_qhparams(self.qscale, {'bitwidth' : 8, 'signed': True})
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth' : 8, 'signed': True}) 
        else : 
            DianaModule.redefine_qhparams(self.qscale, {'bitwidth':new_bitwidth , 'signed': signed})
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth':new_bitwidth , 'signed': signed})
    def clip_scales(self):

        return super().clip_scales()    
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

    def forward(self, input):
        ## Broadcasting to get weights/biases in the correct format 
        broadcasted_weights = self.qscale(self.weights).unsqueeze(1).unsqueeze(1).expand(input.size())
        broadcasted_biases = self.qidentity(self.biases).unsqueeze(1).unsqueeze(1).expand(input.size())
    
        return input * broadcasted_weights+ broadcasted_biases



# pooling layer use regular pool and then have it go through Qidentity 
class DQAvgPool2D(nn.Module, DianaModule): 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,kernel_size, stride=None, padding=0) : 
        super().__init__() 
        self.qidentity = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size,stride = stride, padding=padding) # TODO: Implement the custom avgpool 
    def start_observing(self):
        self.qidentity.start_observing()
    def stop_observing(self):
        self.qidentity.stop_observing()
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if not self.qidentity._is_quantised:  # should only be called after stop_observing() 
            # not quantized 
            return 
        if HW_Behaviour: #defaults hardware parameters ternary behaviour
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth' : 8, 'signed': True}) 
        else : 
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth':new_bitwidth , 'signed': signed})
    def clip_scales(self):
        return super().clip_scales()
    def forward(self, input) : 
        out = self.avgpool(input)
        out = self.qidentity(out) 
        return out 



class DQFC(nn.Module, DianaModule): # output quantized #TODO do I Quantized bias ? 
    def __init__(self,qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType , in_features: int , out_features: int): 
        self.qlinear = QLinear(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features, bias=False)
        self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_features) -0.5)) # think of another way to initialise the biases 
        self.qidentity = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
        self.qout = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
    def start_observing(self) : 
        self.qidentity.start_observing() 
        self.qlinear.start_observing() 
        self.qout.start_observing()
    def stop_observing(self) :
        self.qidentity.stop_observing()  
        self.qlinear.stop_observing()
        self.qout.start_observing()
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if not self.qlinear._is_quantised or not self.qidentity._is_quantised or not self.qout._is_identity:  # should only be called after stop_observing() 
            # not quantized 
            return 
        if HW_Behaviour: #defaults hardware parameters ternary behaviour
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth' : 8, 'signed': True}) 
            DianaModule.redefine_qhparams(self.qlinear, {'bitwidth' : 8, 'signed': True}) 
            DianaModule.redefine_qhparams(self.qout, {'bitwidth' : 8, 'signed': True}) 
        else : 
            DianaModule.redefine_qhparams(self.qidentity, {'bitwidth' : new_bitwidth, 'signed': signed}) 
            DianaModule.redefine_qhparams(self.qlinear, {'bitwidth' : new_bitwidth, 'signed': signed}) 
            DianaModule.redefine_qhparams(self.qout, {'bitwidth' : new_bitwidth, 'signed': signed}) 
    def clip_scales(self):
        return super().clip_scales()
    def forward(self, input : torch.Tensor): 
        broadcasted_input = input.flatten(start_dim=1) 
        broadcasted_biases = self.qidentity(self.biases).expand(input.size(0),)# broadcasted for batches 
        return self.qout(self.qlinear(broadcasted_input)  + broadcasted_biases) # make sure input sizes match 


class DIConvLayer(nn.Module): # default bias false , implemented withoutput quantizer. 
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
        pass 
        self.qin = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
        self.qconv = QConv2d(qrangespec=qrangespec,qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec,kernel_size=kernel_size,stride=stride, padding=padding,in_channels=in_channels ,out_channels=out_channels, dilation=dilation, groups=groups, bias = bias) 
        self.qout = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec)
        self.bias_enabled = bias 
        if self.bias_enabled: 
            self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_channels) -0.5)) # think of another way to initialise the biases 
            self.qbiasin = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
    def start_observing(self): 
        self.qin.start_observing()
        self.qconv.start_observing() 
        if self.bias_enabled: 
            self.qbiasin.start_observing()
    def stop_observing(self): 
        self.qin.stop_observing()
        self.qconv.stop_observing() 
        if self.bias_enabled: 
            self.qbiasin.stop_observing()

    def forward(self, input): 
        broadcasted_bias = 0 
        if self.bias_enabled: # still need to initialize the bias correctly. Right now it's not initialized correctly 
            broadcasted_bias = self.qbiasin(self.biases)
        output = self.qout(self.qconv(self.qin(input)) + broadcasted_bias)
        return output

#In training scales are already matched no custom res_add is needed, but when doing inference scales need to be accounted for 



