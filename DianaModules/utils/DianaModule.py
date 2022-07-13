#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor

from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
from torch import nn 

class DianaModule(nn.Module): # Module to take care of iterative training 
    def start_observing(self): #before starting training with FP 
        for _ ,module in enumerate(self.modules()):
            if issubclass(type(module), _QModule) or (issubclass(type(module),DianaModule) and self is not module) : 
                
                module.start_observing()

    def stop_observing(self): # before starting training with fake quantised network  
        for _ ,module in enumerate(self.modules()):
            if issubclass(type(module), _QModule) or (issubclass(type(module),DianaModule) and self is not module) : 
                module.stop_observing()
                 
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): # before mapping scale and retraining # TODO change from new_bitwdith and signed to list of qrangespecs or an arbitary number of kwqrgs  
        for _ ,module in enumerate(self.modules()): 
            if (issubclass(type(module), _QModule) and issubclass(type(module), DianaBaseOperation)  and module._is_quantised) or(issubclass(type(module),DianaModule) and self is not module) : 
                module.map_scales(new_bitwidth=new_bitwidth, signed = signed, HW_Behaviour=HW_Behaviour)
    def clip_scales(self): # Now it's clipping scales to the nearest power of 2 , but for example if the qhinitparamstart is mean std we can check the nearest power of 2 that would contain a distribution range of values that we find acceptable 
        for _ ,module in enumerate(self.modules()):  
            if issubclass(type(module), _QModule) and module._is_quantised: 
                module.scale = torch.tensor(2**floor(log2(module.scale))) 
            elif (issubclass(type(module),DianaModule) and self is not module) : 
                module.clip_scales() # recursion

class DianaBaseOperation():   
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): # before mapping scale and retraining 
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



