
#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from math import log2, floor
from random import randint
from typing import Any, Dict, List, Union

import matplotlib as plt
from numpy import isin 

from DianaModules.core.Operations import DianaBaseOperation
from DianaModules.utils.onnx import DianaExporter
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd
from .Editing import DianaF2FConverter, DianaF2TConverter

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
import torch.fx as fx
from torch import nn, rand
import quantlib.editing.graphs as qg
import quantlib.backends as qb 
from torch import optim 
import torch.utils.data as ut 
from  torch.utils.data import Dataset as ds  


class DianaModule: # Base class for all diana models  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self,graph_module: Union[nn.Module, fx.graph_module.GraphModule ] ): 
     
        self.gmodule = graph_module
        self._integrized = False 
        self.train_dataset = {} 
        self.validation_dataset = {}
    def start_observing(self): #before starting training with FP 
        for _ ,module in self.gmodule.named_modules():
            if isinstance(module,( _QModule,HarmonisedAdd)) : 
               
                module.start_observing()

    def stop_observing(self): # before starting training with fake quantised network  
        for _ ,module in self.gmodule.named_modules():
            if isinstance(module,_QModule) : 
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
    def named_modules(self): 
        return self.gmodule.named_modules()
    
    @classmethod 
    def from_fp_model(cls, model: nn.Module): # from_ floating point quantised model 
        modulewisedescriptionspec = ( # change const later
            ({'types': ('Identity')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), 
            ({'types': ('ReLU')} , ('per-array' , {'bitwidth': 7, 'signed': False} , ('const', {'a': 0.0 ,'b': 200}) , 'DIANA')), # upper clip is updated during training
            ({'types': ('Linear', 'Conv2d' )}, ('per-array', {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), # can use per-outchannel_weights here  #TODO In the future , test with per-channel quantization and ask compiler team if it's possible to that 
        )
            
        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'meanstd', 'DIANA')
        addtreeforceoutputeps =  False # set to false because each module quantizes the input differently 
        graph = qg.fx.quantlib_symbolic_trace(root=model) # graph module 

        converter  =  DianaF2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            addtreeforceoutputeps
        )
      
        converted_graph =converter(graph) 
        
        return DianaModule(converted_graph)
         

    def true_quantize(self, custom_editors : List[Editor]=[]): # integrise model 
        converter = DianaF2TConverter(custom_editor=custom_editors)
        x, _ = self.train_dataset['dataset'].__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        self.gmodule = converter(self.gmodule, {'x': {'shape': x.shape, 'scale':self.train_dataset['scale']}})
        self._integrized = True
        pass 

    def export_model(self): # x is an integrised tensor input is needed for validation in dory graph 
        if not self._integrized: 
            raise NotImplementedError

        exporter = DianaExporter()
        from pathlib import Path
       
        data_folder = Path("backend/test")
        x, _ = self.validation_dataset['dataset'].__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        x = (x / self.validation_dataset['scale'] ). floor() #integrize 
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
    

    def attach_train_dataset(self, dataset: ut.Dataset , scale : torch.Tensor = torch.Tensor([1])):        
        self.train_dataset['scale'] = scale 
        self.train_dataset['dataset'] = dataset
        self.train_dataset['size'] = len(dataset)

    def attach_validation_dataset(self, dataset: ut.Dataset , scale : torch.Tensor = torch.Tensor([1])): 
        self.validation_dataset['scale'] = scale 
        self.validation_dataset['dataset'] = dataset
        self.validation_dataset['size'] = len(dataset ) 
      
    @classmethod
    def train(cls, model: nn.Module, optimizer , data_loader : Dict[str, ut.DataLoader ], epochs = 100 , criterion = nn.CrossEntropyLoss() , scheduler: Union[None, optim.lr_scheduler._LRScheduler]=None  ): 
        metrics = {'train': {'loss': [], 'acc': []} , 'validate': {'loss': [], 'acc': []} }
        
        assert (model and optimizer) 
        for e in range(epochs):     
            for stage in ['train', 'validate']: 
                running_loss = 0 
                running_correct = 0 
                if stage == 'train': 
                    model.train() 
                else : 
                    model.eval () 
            
                for i,data in enumerate(data_loader[stage]): 
                    x , y = data[0].to(cls.device), data[1].to(cls.device)
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

    def QA_iterative_train  (self, criterion = nn.CrossEntropyLoss() , scheduler: Union[None , optim.lr_scheduler._LRScheduler]=None,  epochs = 1000 , batch_size = 64 , output_weights_path : Union[str ,None] = None , train_FP_model : bool = True ,train_8bit_model : bool = True , train_HWmapped_model : bool = True, train_HW_model : bool = True): 
        self.gmodule.to(DianaModule.device)
        data_loader = {'train': ut.DataLoader(self.train_dataset['dataset'], batch_size=batch_size) , 'validate' : ut.DataLoader(self.validation_dataset['dataset'], batch_size=floor(batch_size/self.train_dataset['size'] * self.validation_dataset['size']))}
        #Iteration 1 - FP Training & pass quantisation specs of 8-bit to model 
        if train_FP_model:  
            optimizer = optim.SGD(self.gmodule.parameters() , lr = 0.01 , momentum = 0.9)  
            FP_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
            if output_weights_path is not None : 
                out_path = output_weights_path + self.gmodule._get_name()+'_FPweights.pth' 
                torch.save(self.gmodule.state_dict(), out_path)
            #DianaModule.plot_training_metrics(FP_metrics) 
        #Iteration 2 - Fake Quantistion all to 8 bit 
        self.start_observing()
        # put 100 validation data sample through and initialize quantization hyperparameters 
        
        for i in range(100): 
            idx  = randint(0 , self.validation_dataset['size'] -1 )
            x, _ = self.validation_dataset['dataset'].__getitem__(idx)
            if len(x.shape) == 3 : 
                x = x.unsqueeze(0)
            x = x/self.validation_dataset['scale'] 
            _ = self.gmodule(x) 

        self.stop_observing() 
        if train_8bit_model: 
            q8b_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
            if output_weights_path is not None : 
                out_path = output_weights_path + self.gmodule._get_name()+'_FQ8weights.pth' 
                torch.save(self.gmodule.state_dict(), out_path)
            #DianaModule.plot_metrics(q8b_metrics ) 
        #Iteration 3 - Input HW specific quantisation, map scales 
        self.map_scales(HW_behaviour=True) 
        if train_HWmapped_model:  
            qHW_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
            if output_weights_path is not None : 
                out_path = output_weights_path + self.gmodule._get_name()+'_FQDianaweights.pth' 
                torch.save(self.gmodule.state_dict(), out_path)
            #DianaModule.plot_metrics(qHW_metrics) 
        #Iteration 4 - clip scales to the power of 2 #TODO Enable noise nodes and retrain 
        self.clip_scales() 
        if train_HW_model: 
            qSc_metrics =  DianaModule.train(self.gmodule, optimizer,data_loader, epochs, criterion, scheduler )
            if output_weights_path is not None : 
                out_path = output_weights_path + self.gmodule._get_name()+'_FQHWweights.pth' 
                torch.save(self.gmodule.state_dict(), out_path)
            #DianaModule.plot_metrics(qSc_metrics ) 



# - In the integrised model pass the relu clipping as a scale and then divide by scale in following conv module
# write your own applier that is added in the end to replace first conv layer with digital conv and replace harmonise adds with a correct dianamodule 
# Could write custom harmoniser or edit the harmoniser adds in the true quant step 


