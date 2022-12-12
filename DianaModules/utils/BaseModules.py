
#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from cProfile import run
from collections import OrderedDict
from math import log2, floor
from pathlib import Path
from random import randint
import sched
from typing import Any, Dict, List, Union

from DianaModules.core.Operations import AnalogConv2d, AnalogOutIdentity, DIANAConv2d, DIANAIdentity, DIANALinear, DIANAReLU, DianaBaseOperation
from DianaModules.utils.converters.fake2true import LayerIntegrizationConverter
from DianaModules.utils.converters.float2fake import F2FConverter
from DianaModules.utils.converters.hwquantization import HWMappingConverter
from DianaModules.utils.onnx import DianaExporter
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qmodules import _PACTActivation
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd

from quantlib.algorithms.qmodules.qmodules.qmodules import _QActivation, _QModule
import torch 
import torch.fx as fx
from torch import nn, rand
import quantlib.editing.graphs as qg
import quantlib.backends as qb 
from torch import optim 
import torch.utils.data as ut 
from  torch.utils.data import Dataset as ds  
import importlib
import pytorch_lightning as pl 
import torchmetrics 
class DianaModule(pl.LightningModule): # Base class for all diana models  
    def __init__(self,graph_module: Union[nn.Module, fx.graph_module.GraphModule ]=None, criterion= nn.CrossEntropyLoss()): 
        super().__init__()
        self.gmodule = graph_module
        self._integrized = False 
        self.train_dataloader = {} 
        self.validation_dataloader = {}
        self.quant_dataloader  = {} 
        self.optimizer = None 
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        #self.test_acc = torchmetrics.Accuracy() 
        self.criterion = criterion
    
    def start_observing(self , is_modules = (_QModule, HarmonisedAdd), not_modules = type(None)): #before starting training with FP 
        for _ ,module in self.gmodule.named_modules(): 
            if isinstance(module,is_modules) and not isinstance(module ,not_modules) : 
                module.start_observing()

    def stop_observing(self, is_modules = (_QModule, HarmonisedAdd), not_modules = type(None)): # before starting training with fake quantised network  
        for _ ,module in self.gmodule.named_modules():
            if isinstance(module,is_modules) and not isinstance(module , not_modules): 
                module.stop_observing()
   
    def forward(self, x : torch.Tensor) : 
        return self.gmodule(x)
    
    def freeze_clipping_bound(self): 
        for _ , module in self.gmodule.named_modules(): 
            if isinstance(module ,DIANAReLU) : 
                module.freeze()
    def unfreeze_clipping_bound(self) -> None:
        for _ , module in self.gmodule.named_modules(): 
            if isinstance(module ,DIANAReLU) : 
                module.thaw()
    @classmethod 
    def from_trainedfp_model(cls, model: nn.Module , modules_descriptors  = None): # returns fake quantized model from_ floating point quantised model 
        modulewisedescriptionspec = ( # change const later
            ({'types': ('Identity')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), 
            ({'types': ('ReLU')} , ('per-array' , {'bitwidth': 7, 'signed': False} ,'meanstd' , 'DIANA')), # upper clip is updated during training
            ({'types': ( 'Conv2d' )}, ('per-outchannel_weights',{'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), # can use per-outchannel_weights here 
            ({'types': ( 'Linear' )}, ('per-outchannel_weights', {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')),
        )
        analogcoredescriptionspec =  ('per-array', 'ternary' , 'meanstd' )
            
        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'meanstd', 'DIANA'  )  
        graph = qg.fx.quantlib_symbolic_trace(root=model) # graph module 
       
        converter  =  F2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            analogcoredescriptionspec , 
            modules_descriptors=modules_descriptors
         
        )
      
        converted_graph =converter(graph) 
        
        return converted_graph
    
    def map_to_hw(self , custom_editors : List[Editor]=[])  : 
        # free relus upper boun d
        
        for _ , mod in self.named_modules(): 
            if isinstance(mod , DIANAReLU) : 
                mod.freeze() 
        converter = HWMappingConverter(custom_editor=custom_editors)
        x, _ = self.train_dataloader['dataloader'].dataset.__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        with torch.no_grad(): 
            self.gmodule = converter(self.gmodule, {'x': {'shape': x.shape, 'scale':self.train_dataloader['scale']}}, input=x)
        
    def integrize_layers(self)    : 
        
        converter = LayerIntegrizationConverter()
        x, _ = self.train_dataloader['dataloader'].dataset.__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        x.to("cpu")
        with torch.no_grad(): 
            self.gmodule = converter(self.gmodule, {'x': {'shape': x.shape, 'scale':self.train_dataloader['scale']}})
        self._integrized = True
        pass 

    def export_model(self, data_folder : str): # x is an integrised tensor input is needed for validation in dory graph 
        if not self._integrized: 
            raise NotImplementedError

        exporter = DianaExporter()
        from pathlib import Path
       
        x, _ = self.train_dataloader['dataloader'].dataset.__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        x = (x / self.train_dataloader['scale'] ). floor() #integrize 
        exporter.export(network=self.gmodule, input_shape=x.shape, path=data_folder)
        exporter.dump_features(network=self.gmodule, x=x, path=data_folder ) 
        pass 
    
    def attach_train_dataloader(self, dataloader, scale : torch.Tensor = torch.Tensor([1])):        
        self.train_dataloader['scale'] = scale 
        self.train_dataloader['dataloader'] = dataloader 
        self.train_dataloader['size'] = len(dataloader.dataset) 

    def attach_validation_dataloader(self, dataloader , scale : torch.Tensor = torch.Tensor([1])): 
        self.validation_dataloader['scale'] = scale 
        self.validation_dataloader['dataloader'] = dataloader 
        self.validation_dataloader['size'] = len(dataloader.dataset) 

    def attach_quantization_dataloader(self, dataloader ): 
        self.quant_dataloader['dataloader'] = dataloader 
        self.quant_dataloader['size'] = len(dataloader.dataset) 
         

    def training_step(self, batch, batch_idx, *args, **kwargs) :
        x , y = batch  
        if self._integrized: 
            x = torch.floor(x / self.train_dataloader["scale"].to(x.device))  
        yhat = self.gmodule(x)
        loss = self.criterion(yhat , y)
        self.log("train_loss", loss , prog_bar=True)
        # acc 
        return {"loss": loss, "pred":yhat, "true":y}
    def training_step_end(self, outputs) : 

        self.train_acc(outputs["pred"] ,outputs["true"] )
        self.log("train_acc" , self.train_acc ,on_step=False,  on_epoch=True, prog_bar=True, sync_dist=True)

        return outputs["loss"]
        
    def test_step(self, batch, batch_idx, *args, **kwargs) : # for quantization
        
        x , y = batch 
        _ = self.gmodule(x) 
     
    def validation_step(self, batch, batch_idx, *args, **kwargs) :
        x , y = batch  
        #if self._integrized: 
        #    x = torch.floor(x / self.train_dataloader["scale"].to(x.device)) 
        yhat = self.gmodule(x)

        loss = self.criterion(yhat , y)
        
        
        
        self.log("val_loss", loss , prog_bar=True ,sync_dist=True)
        return {"loss": loss, "pred":yhat, "true":y}
        
    def validation_step_end(self, outputs) :
        self.valid_acc(outputs["pred"] ,outputs["true"] )
        self.log("val_acc" ,self.valid_acc,  on_epoch=True , prog_bar=True, sync_dist=True)
        
        return outputs["loss"]
    def set_optimizer(self, type : str = 'SGD' , *args , **kwargs):  # case sensitive 
        my_module = importlib.import_module("torch.optim")
        MyClass = getattr(my_module, type) 
        self.optimizer = MyClass(self.gmodule.parameters() , *args, **kwargs) 
        self.scheduler  = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , mode='max',patience=5)
    
    def configure_optimizers(self):
        
        return {"optimizer":self.optimizer , "lr_scheduler": {"scheduler": self.scheduler ,"monitor": "val_acc"} } 

    def initialize_quantization(self , trainer): 
        self.start_observing()
        trainer.test(model=self , dataloaders=self.quant_dataloader['dataloader']) 
        self.stop_observing() 
        for _,module in self.gmodule.named_modules(): 
            if type(module) == DIANAConv2d or isinstance(module, DIANALinear) or (isinstance(module, _QActivation) and not isinstance(module, AnalogOutIdentity) ):  
                module.scale = torch.Tensor(torch.exp2(torch.round(torch.log2(module.scale))) )     
    def initialize_quantization_activations(self, trainer): 
        self.start_observing(_QActivation)
        trainer.test(model=self , dataloaders=self.quant_dataloader['dataloader'])  
        self.stop_observing(_QActivation)
    def initialize_quantization_layers(self, trainer): 
        self.start_observing(not_modules=_QActivation)
        trainer.test(model=self , dataloaders=self.quant_dataloader['dataloader']) 
        self.stop_observing(not_modules=_QActivation)
    def set_quantized(self, activations=True):   
        x ,_ = self.train_dataloader["dataloader"].dataset.__getitem__(0) 
        if len(x.shape <4): 
            x = x.unsqueeze(0) 
        if activations: 
            self.start_observing()
        else: 
            self.start_observing(not_modules=_QActivation)
        _ = self.gmodule(x) 
        if activations: 
            self.stop_observing()
        else: 
            self.stop_observing(not_modules=_QActivation)
    @classmethod
    def return_DFS (cls,  node : fx.node.Node,  depth : int)  :
        ls : List[fx.node.Nodes] = [] 
        n = node
        for i in range(depth) : 
            users = [ u for u in n.users] 
            if len(users) == 0: 
                 return ls
            ls.append(users[0]) 
            n = users[0]
        return ls
    @classmethod
    def remove_data_parallel(cls,old_state_dict):
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
    
        return new_state_dict