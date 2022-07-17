#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor
from .Editing import DianaF2FConverter, DianaF2FQuantiser
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qmodules import _PACTModule

from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
from torch import nn
import quantlib.editing.graphs as qg

from quantlib.editing.editing.float2fake import F2FConverter 

class DianaModule(nn.Module): # Module to take care of iterative training 
    def __init__(self): 
        super().__init__() 
        self._integrized = False 
        self.traced_graph = None
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
                if issubclass(type(module), _PACTModule): # Pact activations 
                    module.clip_hi =  torch.tensor(2**floor(log2(module.clip_hi))) 
                    
            elif (issubclass(type(module),DianaModule) and self is not module) : 
                module.clip_scales() # recursion

    @classmethod 
    def fquantize_model8bit(cls, model: nn.Module) -> F2FConverter: # from_ fake quantised model 
        modulewisedescriptionspec = (
            ({'types': ('Identity','ReLU')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'minmax','DIANA')), 
            ({'types': ('Linear', 'Conv2d' , 'BatchNorm2d')}, ('per-array', {'bitwidth': 8, 'signed': True},  'minmax','DIANA')), # can use per-outchannel here 
        )
            
        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'minmax', 'DIANA')
        addtreeforceoutputeps = True
        graph = qg.fx.quantlib_symbolic_trace(root=model) # graph module 
         
        
        converter  =  DianaF2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            addtreeforceoutputeps
         )
      

        return converter(graph)
         

    def true_quantize(self): # integrise model 

        self._integrized = True
        pass 

    def export_model(self): 
        if not self._integrized: 
            # error 
            pass 
        pass 
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



