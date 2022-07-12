#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod

from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec,create_qhparams

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 

class DianaModule(): 
    @abstractmethod
    def start_observing(self): #before starting training with FP 
        pass 
    @abstractmethod
    def stop_observing(self): # before starting training with fake quantised network  
        pass 
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): # before mapping scale and retraining 
        pass 
    @abstractmethod 
    def clip_scales(self): # clip scales to closest power of 2  
        pass 
    @classmethod
    def redefine_qhparams(cls, qlayer, qrangespec:               QRangeSpecType):  
        assert(issubclass(qlayer, _QModule))
        qlayer._qrange = resolve_qrangespec(qrangespec)
        zero, n_levels, step, scale = create_qhparams(qlayer._qrange)
        qlayer.zero =  torch.tile(zero,     qlayer._observer.broadcasting_shape)
        qlayer.n_levels=  torch.tile(n_levels,     qlayer._observer.broadcasting_shape)
        qlayer.step=  torch.tile(step,    qlayer._observer.broadcasting_shape)
        qlayer.scale =  torch.tile(scale,   qlayer._observer.broadcasting_shape)
        qlayer._init_qhparams() # reinitialize scale and zero 



