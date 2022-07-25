#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor
from typing import Any, Dict, Union

import matplotlib as plt 

from DianaModules.core.operations import DianaBaseOperation
from DianaModules.utils.onnx import DianaExporter
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from .Editing import DianaF2FConverter, DianaF2TConverter

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
import torch.fx as fx
from torch import nn
import quantlib.editing.graphs as qg
import quantlib.backends as qb 
from torch import optim 
import torch.utils.data as ut 
from  torch.utils.data import Dataset as ds  

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
            if (issubclass(type(module), _QModule) and issubclass(type(module), DianaBaseOperation)  and module._is_quantised) : 
                module.map_scales(new_bitwidth=new_bitwidth, signed = signed, HW_Behaviour=HW_Behaviour)
    def clip_scales(self): # Now it's clipping scales to the nearest power of 2 , but for example if the qhinitparamstart is mean std we can check the nearest power of 2 that would contain a distribution range of values that we find acceptable 
        for _ ,module in enumerate(self.gmodule.modules()):  
            if issubclass(type(module), _QModule) and module._is_quantised: 
                module.scale = torch.tensor(2**floor(log2(module.scale))) 
               
                    
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
            ({'types': ('Identity')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'const','DIANA')), 
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
        self.gmodule = QuantLibRetracer()(converter(self.gmodule, {'x': {'shape': x.unsqueeze(0).shape, 'scale':torch.tensor([ 1])}})) 
        self._integrized = True
        pass 

    def export_model(self, x : torch.Tensor): # x is an integrised tensor input is needed for validation in dory graph 
        if not self._integrized: 
            raise NotImplementedError

        exporter = DianaExporter()
        from pathlib import Path
       
        data_folder = Path("backend/test")

        exporter.export(network=self.gmodule, input_shape=x.shape, path=data_folder.absolute())
        exporter.dump_features(network=self.gmodule, x=x, path=data_folder.absolute() ) 
        pass 
    
    @classmethod
    def plot_training_metrics(metrics : Dict [str , Dict[str, list]]) : 
        fig, (plot1, plot2) = plt.subplots(nrows=1, ncols=2)
        plot1.plot(metrics['train']['loss'], label='train loss')
        plot1.plot(metrics['validate']['loss'], label='val loss')
        lines, labels = plot1.get_legend_handles_labels()
        plot1.legend(lines, labels, loc='best')

        plot2.plot(metrics['train']['acc'], label='train acc')
        plot2.plot(metrics['validate']['acc'], label='val acc')
        plot2.legend()
    pass

    @classmethod
    def train(model: nn.Module, optimizer , data_loader : Dict[str, ut.DataLoader ], epochs = 100 , criterion = nn.CrossEntropyLoss() , scheduler: Union[None, optim.lr_scheduler._LRScheduler]=None): 
        metrics = {}
     
        assert (model and optimizer) 
        for e in range(epochs):     
            for stage in ['train', 'validate']: 
                running_loss = 0 
                running_correct = 0 
                if stage == 'train': 
                    model.train() 
                else : 
                    model.eval () 
            
                for x,y in data_loader[stage]: 
                    optimizer.zero_grad() 
                    if stage == 'validate': 
                        with torch.no_grad(): 
                            yhat = model(x)
                    else: 
                        yhat = model(x) 
                    loss = criterion(yhat, y) 
                    predictions = torch.argmax(yhat)
                    if stage == 'train': 
                        loss.backward() 
                        optimizer.step() 
          
                    running_loss += loss.item() *x.size(0) 
                    running_correct += torch.sum(predictions==y.data).item() # for classification problems
                e_loss = running_loss / len(data_loader[stage].dataset)
                e_acc = running_correct / len(data_loader[stage].dataset)
                print(f'Epoch {e+1} \t\t {stage}ing... Loss: {e_loss} Accuracy :{e_acc}')
            if stage == 'train' and scheduler is not None: 
                scheduler.step()
            metrics[stage]['loss'].append(e_loss)
            metrics[stage]['acc'].append(e_acc)
        return metrics  

    def train_model  (self, optimizer,train_dataset: ds, validation_dataset: ds , criterion = nn.CrossEntropyLoss() , scheduler: Union[None , optim.lr_scheduler._LRScheduler]=None,  epochs = 1000 , batch_size = 64 ): 
        
        data_loader = {'train': ut.DataLoader(train_dataset, batch_size=batch_size) , 'validate' : ut.DataLoader(validation_dataset, batch_size=floor(batch_size/len(train_dataset) * len(validation_dataset)))}
        #Iteration 1 - FP Training & pass quantisation specs of 8-bit to model 
        self.start_observing()
        FP_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
        DianaModule.plot_training_metrics(FP_metrics) 
        #Iteration 2 - Fake Quantistion all to 8 bit 
        self.stop_observing() 
        q8b_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
        DianaModule.plot_metrics(q8b_metrics ) 
        #Iteration 3 - Input HW specific quantisation, map scales 
        self.map_scales(HW_behaviour=True) 
        qHW_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
        DianaModule.plot_metrics(qHW_metrics) 
        #Iteration 4 - clip scales to the power of 2 again and train 
        self.clip_scales() 
        qSc_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
        DianaModule.plot_metrics(qSc_metrics ) 

        
        pass

# - In the integrised model pass the relu clipping as a scale and then divide by scale in following conv module
# write your own applier that is added in the end to replace first conv layer with digital conv and replace harmonise adds with a correct dianamodule 
# Could write custom harmoniser or edit the harmoniser adds in the true quant step 


