
#Abstract class which all diana specific module have to inhereit from 
from abc import abstractmethod
from cProfile import run
from math import log2, floor
from random import randint
from typing import Any, Dict, List, Union

import matplotlib as plt
from numpy import isin 

from DianaModules.core.Operations import AnalogConv2d, DIANAReLU, DianaBaseOperation
from DianaModules.utils.converters.fake2true import LayerIntegrizationConverter
from DianaModules.utils.converters.float2fake import F2FConverter
from DianaModules.utils.converters.hwquantization import HWMappingConverter
from DianaModules.utils.onnx import DianaExporter
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.graphs.nn.harmonisedadd import HarmonisedAdd
from .Converters import DianaF2FConverter, DianaF2TConverter

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
import torch 
import torch.fx as fx
from torch import nn, rand
import quantlib.editing.graphs as qg
import quantlib.backends as qb 
from torch import optim 
import torch.utils.data as ut 
from  torch.utils.data import Dataset as ds  

import importlib


class DianaModule: # Base class for all diana models  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self,graph_module: Union[nn.Module, fx.graph_module.GraphModule ] ): 
        graph_module.to(DianaModule.device) 
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
    def clip_scales_pow2(self): # Now it's clipping scales to the nearest power of 2 , but for example if the qhinitparamstart is mean std we can check the nearest power of 2 that would contain a distribution range of values that we find acceptable 
        for _ ,module in enumerate(self.gmodule.modules()):  
            if issubclass(type(module), _QModule) and  not isinstance(module, AnalogConv2d) and module._is_quantised: 
                module.scale = torch.Tensor(torch.exp2(torch.round(torch.log2(module.scale))) )     
               
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
    def from_trained_model(cls, model: nn.Module , map_to_analog = True ): # returns fake quantized model from_ floating point quantised model 
        modulewisedescriptionspec = ( # change const later
            ({'types': ('Identity')},                             ('per-array',  {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), 
            ({'types': ('ReLU')} , ('per-array' , {'bitwidth': 8, 'signed': False} ,'meanstd' , 'DIANA')), # upper clip is updated during training
            ({'types': ( 'Conv2d' )}, ('per-array', {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')), # can use per-outchannel_weights here  #TODO In the future , test with per-channel quantization and ask compiler team if it's possible to that 
            ({'types': ( 'Linear' )}, ('per-array', {'bitwidth': 8, 'signed': True},  'meanstd','DIANA')),
        )
        analogcoredescriptionspec =  ('per-array', {'bitwidth': 8, 'signed': True} , 'meanstd')
            
        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'meanstd', 'DIANA')
        graph = qg.fx.quantlib_symbolic_trace(root=model) # graph module 

        converter  =  F2FConverter(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            analogcoredescriptionspec , 
            map_to_analog=map_to_analog
        )
      
        converted_graph =converter(graph) 
        
        return converted_graph
    
    def map_to_hw(self , custom_editors : List[Editor]=[])  : 
        # free relus upper boun d
        for _ ,module in self.gmodule.named_modules():
            if isinstance(module,DIANAReLU) : 
                module.freeze()
        #self.clip_scales_pow2()
        converter = HWMappingConverter(custom_editor=custom_editors)
        x, _ = self.train_dataset['dataset'].__getitem__(0)
        if len(x.shape) == 3 : 
            x = x.unsqueeze(0)
        
        self.gmodule = converter(self.gmodule, {'x': {'shape': x.shape, 'scale':self.train_dataset['scale']}})
        self._integrized = True
    def integrize_layers(self)    : 
        for _ ,module in self.gmodule.named_modules():
            if isinstance(module,DIANAReLU) : 
                module.freeze()
        converter = LayerIntegrizationConverter()
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
    def train(cls, model: nn.Module, optimizer , data_loader : Dict[str, ut.DataLoader ], epochs = 100 , criterion = nn.CrossEntropyLoss() , scheduler: Union[None, optim.lr_scheduler._LRScheduler]=None , model_save_path : str = None ): 
        metrics = {'train': {'loss': [], 'acc': []} , 'validate': {'loss': [], 'acc': []} }
        max_val_acc = 0 
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
                            loss = criterion(yhat, y) 
                        
                    else: 
                        with torch.set_grad_enabled(True):
                            yhat = model(x) 
                
                            loss = criterion(yhat, y) 
                            loss.backward() 
                      
                            optimizer.step() 
                    
                    predictions = torch.argmax(yhat , 1)

                    running_loss += loss.item() *x.size(0) 
                    running_correct += torch.sum(predictions==y) .item()
                    
         
                e_loss = running_loss / len(data_loader[stage].dataset)
                e_acc = running_correct / len(data_loader[stage].dataset) 
                print(f'Epoch {e+1} \t\t {stage} stage... Loss: {e_loss:.4f} Accuracy :{e_acc:.4f}')
                if stage == 'train' and scheduler is not None: 
                    scheduler.step()
                metrics[stage]['loss'].append(e_loss)
                metrics[stage]['acc'].append(e_acc)
                if stage == 'validate' and e_acc > max_val_acc and model_save_path is not None: 
                    # save best state dict acc model on 
                    torch.save ({
                        'epoch': e,
                        'state_dict': model.state_dict(),
                        'loss': e_loss,
                        'acc' : e_acc 
                    } , model_save_path)
                    max_val_acc = e_acc 
                    pass 
  
        return metrics  

    def QA_iterative_train  (self, criterion = nn.CrossEntropyLoss() , scheduler: Union[None , optim.lr_scheduler._LRScheduler]=None,  epochs = 100 , batch_size = 128 , output_weights_path : Union[str ,None] = None , train_FP_model : bool = True ,train_8bit_model : bool = True , train_HWmapped_model : bool = True, train_HW_model : bool = True): # example of workflow
        data_loader = {'train': ut.DataLoader(self.train_dataset['dataset'], batch_size=batch_size, shuffle=True, pin_memory=True) , 'validate' : ut.DataLoader(self.validation_dataset['dataset'], batch_size=batch_size, shuffle=True  ,pin_memory=True)}
        #Iteration 1 - FP Training & pass quantisation specs of 8-bit to model 

        
        if train_FP_model:  
            print("Training FP Model...")
            out_path = output_weights_path +"/"+ self.gmodule._get_name()+'_FPweights.pth' if output_weights_path is not None else None
            self.configure_optimizer('SGD' ,lr = 0.01 , momentum = 0.1, weight_decay=5e-5)
            FP_metrics =  DianaModule.train(self.gmodule, self.optimizer,data_loader, epochs, criterion, scheduler, model_save_path=out_path )
            print("Finished Training FP Model...")
            #DianaModule.plot_training_metrics(FP_metrics) 
                 
        #Iteration 2 - Fake Quantistion all to 8 bit 
        self.gmodule = DianaModule.from_trained_model(self.gmodule) #f2f 
        self.initialize_quantization() 
        self.gmodule.to(DianaModule.device)
        if train_8bit_model: 
            print("Training 8bit Model...")
            self.configure_optimizer('SGD', lr = 0.04 , momentum=0.9,  weight_decay=1e-5)
            out_path = output_weights_path + "/"+self.gmodule._get_name()+'_FQ8weights.pth' if output_weights_path is not None else None
            q8b_metrics =  DianaModule.train(self.gmodule, self.optimizer,data_loader, epochs, criterion, scheduler, model_save_path=out_path )
            print("Finished Training 8bit Model...")
            #DianaModule.plot_metrics(q8b_metrics ) 
            
        #Iteration 3 - Input HW specific quantisation, map scales 
        self.map_scales(HW_Behaviour=True)
      
        if train_HWmapped_model:  
            print("Training HW_Mapped Model...")
           
            self.configure_optimizer( lr = 0.04 , momentum=0.9,  weight_decay=1e-5 )
            out_path = output_weights_path +"/"+ self.gmodule._get_name()+'_FQMappedweights.pth'   if output_weights_path is not None else None
            qHW_metrics =  DianaModule.train(self.gmodule, self.optimizer,data_loader, epochs, criterion, scheduler, model_save_path=out_path )
            #DianaModule.plot_metrics(qHW_metrics)
            print("Finished Training HW_Mapped Model...")
     
             
        #Iteration 4 - clip scales to the power of 2 #TODO Enable noise nodes and retrain 
        self.clip_scales_pow2() 
        if train_HW_model: 
            print("Training Final HW Model...")
            self.configure_optimizer( lr = 0.04 , momentum=0.9,  weight_decay=1e-5 )
            out_path = output_weights_path +"/"+ self.gmodule._get_name()+'_FQHWweights.pth' if output_weights_path is not None else None
            qSc_metrics =  DianaModule.train(self.gmodule, self.optimizer,data_loader, epochs, criterion, scheduler, model_save_path=out_path )
            #DianaModule.plot_metrics(qSc_metrics ) 
            print("Finished Final HW Model...")
        

    def configure_optimizer(self, type : str = 'SGD' , *args , **kwargs):  # case sensitive 
        my_module = importlib.import_module("torch.optim")
        MyClass = getattr(my_module, type) 
        self.optimizer = MyClass(self.gmodule.parameters() , *args, **kwargs) 
    
    def initialize_quantization(self, count = 400): 
        print("starting initialization process on " ,DianaModule.device )
        self.start_observing()
        # put count validation data samples through and initialize quantization hyperparameters 
        # do this in CPU 
        for i in range(count ): 
            idx  = randint(0 , self.validation_dataset['size'] -2 )
            x, _ = self.validation_dataset['dataset'].__getitem__(idx) 
    
            if len(x.shape) == 3 : 
                x = x.unsqueeze(0).to(DianaModule.device)
                _ = self.gmodule(x) 
        self.stop_observing() 
       
        

    def evaluate_model(self, criterion=nn.CrossEntropyLoss() , batch_size=128) : 
        self.gmodule.eval () 
        data_loader = {'train': ut.DataLoader(self.train_dataset['dataset'], batch_size=batch_size, shuffle=True, pin_memory=True) , 'validate' : ut.DataLoader(self.validation_dataset['dataset'], batch_size=batch_size, shuffle=True  ,pin_memory=True)}
        running_loss = 0 
        running_correct = 0 
        for i,data in enumerate(data_loader['validate']): 
            
            x , y = data[0].to(DianaModule.device), data[1].to(DianaModule.device)
            if self._integrized: 
                x = torch.floor(x / self.validation_dataset['scale'].to(x.device)) 
            with torch.no_grad(): 
                yhat = self.gmodule(x)
                loss = criterion(yhat, y) 
                
            
            predictions = torch.argmax(yhat , 1)

            running_loss += loss.item() *x.size(0) 
            running_correct += torch.sum(predictions==y) .item()
            
 
        e_loss = running_loss / len(data_loader['validate'].dataset)
        e_acc = running_correct / len(data_loader['validate'].dataset) 
        return e_loss , e_acc 
       