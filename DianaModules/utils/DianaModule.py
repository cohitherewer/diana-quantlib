#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor

from pyrsistent import v
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 

class DianaModule(): 
    def start_observing(self): #before starting training with FP 
        for _, val in self.__dict__: 
            if issubclass(type(val), _QModule): 
                val.start_observing()
    
    def stop_observing(self): # before starting training with fake quantised network  
        for _, val in self.__dict__: 
            if issubclass(type(val), _QModule): 
                val.stop_observing()
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): # before mapping scale and retraining # TODO change from new_bitwdith and signed to list of qrangespecs or an arbitary number of kwqrgs  
        for _, val in self.__dict__: 
            if issubclass(type(val), _QModule) and issubclass(type(val), DianaBaseOperation) and val._is_quantised: 
                val.map_scales(new_bitwidth=new_bitwidth, signed = signed, HW_Behaviour=HW_Behaviour)
    def clip_scales(self): # Now it's clipping scales to the nearest power of 2 , but for example if the qhinitparamstart is mean std we can check the nearest power of 2 that would contain a distribution range of values that we find acceptable 
        for _, val in self.__dict__: 
            if issubclass(type(val), _QModule) and val._is_quantised: 
                val.scale = 2**floor(log2(val.scale))

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



