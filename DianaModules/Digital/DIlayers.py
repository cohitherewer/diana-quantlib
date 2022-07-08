import torch 
from torch import nn
from typing import Union 
import torch.nn.functional as F
from quantlib.algorithms.qmodules.qmodules import  QIdentity
from quantlib.algorithms.qmodules.qmodules.qlinears import QLinear 


from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from Functions._FakeQuantizer import _FakeDQuantiser
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
    
    def _register_qop(self, func): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)  





class DQScaleBias(nn.Module): # ask about this one 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:             torch.Tensor, 
                 input_scale: Union[None, float]): #channels * batch_size  
        self.qscale = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(in_channels) -0.5))
        self.register_parameter(name='weights', param = torch.nn.parameter.Parameter( torch.randn(in_channels)))
        self.qidentity = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
        self.relufunc = nn.ReLU()

    def start_observing(self): 
        self.qscale.start_observing()
        self.qidentity.start_observing() 
  
      #when stopping observations scale can be checked, modified and rounder to closest power of 2 #TODO
      #this is implied   i_scale*w_scale*b_scale > 1 if above criteria is fulfilled 
    def stop_observing(self):
        self.qscale.stop_observing()
        self.qidentity.stop_observing() 
        
       
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
     
        return self.relufunc(input * broadcasted_weights+ broadcasted_biases) 



# pooling layer use regular pool and then have it go through Qidentity 
class DQAvgPool2D(nn.Module): 
    def __init__(self, qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,kernel_size, stride=None, padding=0) : 
        super().__init__() 
        self.qidentity = DQIdentity(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size,stride = stride, padding=padding) # TODO: Implement the custom avgpool 
    
    def forward(self, input) : 
        out = self.avgpool(input)
        out = self.qidentity(out) 
        return out 



class DQFC(nn.Module): 
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
        
        pass 
    def start_observing(self) : 
        self.qidentity.start_observing() 
        self.qlinear.start_observing() 
    def stop_observing(self) :
        self.qidentity.stop_observing()  
        self.qlinear.stop_observing()

    def forward(self, input : torch.Tensor): 
        broadcasted_input = input.flatten(start_dim=1) 
        broadcasted_biases = self.qidentity(self.biases).expand(input.size(0),)
        return self.qlinear(broadcasted_input)  + broadcasted_biases # make sure input sizes match 

class DIConvLayer(nn.Module): # default bias false 
    def __init__(self): 
        pass 
    def start_observing(self): 
        pass
    def stop_observing(self): 
        pass 
    def forward(self, input): 
        pass 

#Abstract class which all diana specific module have to inhereit from 
class DianaModule(): 
    def start_observing(): #before starting training with FP 
        pass 
    def stop_observing(): # before starting training with fake quantised network  
        pass 
    def map_scales(): # before mapping scale and retraining 
        pass 
