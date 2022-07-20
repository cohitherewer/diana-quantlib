#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor
from turtle import forward
from typing import Any, Dict, Union

from DianaModules.core.operations import DianaBaseOperation
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from .Editing import DianaF2FConverter, DianaF2TConverter
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qmodules import _PACTModule

from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec
from quantlib.algorithms.qbase import QRangeSpecType

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
import torch.fx as fx
from torch import nn
import quantlib.editing.graphs as qg
import quantlib.backends as qb 

class DianaModule: # Base class for all diana models  
    def __init__(self,graph_module: fx.graph_module.GraphModule): 
        self.gmodule = graph_module
        self._integrized = False 
       
    def start_observing(self): #before starting training with FP 
        for _ ,module in enumerate(self.gmodule.modules()):
            if issubclass(type(module), _QModule) or (issubclass(type(module),DianaModule) and self is not module) : 
                
                module.start_observing()

    def stop_observing(self): # before starting training with fake quantised network  
        for _ ,module in enumerate(self.gmodule.modules()):
            if issubclass(type(module), _QModule) or (issubclass(type(module),DianaModule) and self is not module) : 
                module.stop_observing()
                 
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): # before mapping scale and retraining # TODO change from new_bitwdith and signed to list of qrangespecs or an arbitary number of kwqrgs  
        for _ ,module in enumerate(self.gmodule.modules()): 
            if (issubclass(type(module), _QModule) and issubclass(type(module), DianaBaseOperation)  and module._is_quantised) or(issubclass(type(module),DianaModule) and self is not module) : 
                module.map_scales(new_bitwidth=new_bitwidth, signed = signed, HW_Behaviour=HW_Behaviour)
    def clip_scales(self): # Now it's clipping scales to the nearest power of 2 , but for example if the qhinitparamstart is mean std we can check the nearest power of 2 that would contain a distribution range of values that we find acceptable 
        for _ ,module in enumerate(self.gmodule.modules()):  
            if issubclass(type(module), _QModule) and module._is_quantised: 
                module.scale = torch.tensor(2**floor(log2(module.scale))) 
                if issubclass(type(module), _PACTModule): # Pact activations 
                    module.clip_hi =  torch.tensor(2**floor(log2(module.clip_hi))) 
                    
            elif (issubclass(type(module),DianaModule) and self is not module) : 
                module.clip_scales() # recursion
    def forward(self, x : torch.Tensor) : 
        return self.gmodule(x)
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)
    def modules(self): 
        return self.gmodule.modules()
    @classmethod 
    def fquantize_model8bit(cls, model: nn.Module): # from_ floating point quantised model 
        modulewisedescriptionspec = ( # change const later
            ({'types': ('Identity','ReLU')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'const','DIANA')), 
            ({'types': ('Linear', 'Conv2d' )}, ('per-array', {'bitwidth': 8, 'signed': True},  'const','DIANA')), # can use per-outchannel here 
        )
            
        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'const', 'DIANA')
        addtreeforceoutputeps = False # set to false because each module quantizes the input differently 
        graph = qg.fx.quantlib_symbolic_trace(root=model) # graph module 

        converter  =  DianaF2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            addtreeforceoutputeps
         )
      
        converted_graph =QuantLibRetracer()(converter(graph) )
        
        return DianaModule(converted_graph)
         

    def true_quantize(self): # integrise model 
        converter = DianaF2TConverter()
        x = torch.rand(3,20,20)
        self.gmodule = QuantLibRetracer()(converter(self.gmodule, {'x': {'shape': x.unsqueeze(0).shape, 'scale':torch.tensor([ 0.020625000819563866])}})) 
        self._integrized = True
        pass 

    def export_model(self, x : torch.Tensor): # x is an integrised tensor input is needed for validation in dory graph 
        if not self._integrized: 
            raise NotImplementedError

        exporter = qb.dory.DORYExporter()
        from pathlib import Path
       
        data_folder = Path("../backend/test")

        exporter.export(network=self.gmodule, input_shape=x.shape, path="data_folder")
        exporter.dump_features(network=self.gmodule, x=x, path="data_folder" ) 
        pass 



# - In the integrised model pass the relu clipping as a scale and then divide by scale in following conv module
# write your own applier that is added in the end to replace first conv layer with digital conv and replace harmonise adds with a correct dianamodule 
# Could write custom harmoniser or edit the harmoniser adds in the true quant step 


