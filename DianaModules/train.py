from math import gamma
from sched import scheduler
from pyparsing import Combine
from sqlalchemy import null
import torch
from torch import nn
import torch.utils.data as ut
from torch import nn , optim
from torchvision import transforms
from torchvision import datasets as ds
from typing import Dict, Union 
from math import floor
import matplotlib as plt

from DianaModules.Models.QResnet20 import QResnet20



def quantization_aware_train(model , optimizer,train_dataset: ds, validation_dataset: ds , criterion = nn.CrossEntropyLoss() , scheduler: Union[None , optim.lr_scheduler._LRScheduler]=None,  epochs = 1000 , batch_size = 64 ): 
    #train_loader = ut.DataLoader(dataset=train_dataset, batch_size=batch_size) 
    
    qrangespec =  {'bitwidth': 8 , 'signed': True} 
    qgranularityspec = 'per-array' 
    qhparamsinitstrategyspec  = 'meanstd'
    data_loader = {'train': ut.DataLoader(train_dataset, batch_size=batch_size) , 'validate' : ut.DataLoader(validation_dataset, batch_size=floor(batch_size/len(train_dataset) * len(validation_dataset)))}
    
 
    #Iteration 1 - FP Training & pass quantisation specs of 8-bit to model 
    model.start_observing()
    FP_metrics =  train(model, optimizer,data_loader, epochs, criterion, scheduler )
    plot_metrics(FP_metrics) 
    #Iteration 2 - Fake Quantistion all to 8 bit 
    model.stop_observing() 
    q8b_metrics =  train(model, optimizer,data_loader, epochs, criterion, scheduler )
    plot_metrics(q8b_metrics ) 
    #Iteration 3 - Input HW specific quantisation, map scales 
    model.map_scales(HW_behaviour=True) 
    qHW_metrics =  train(model, optimizer,data_loader, epochs, criterion, scheduler )
    plot_metrics(qHW_metrics) 
    #Iteration 4 - clip scales to the power of 2 again and train 
    model.clip_scales() 
    qSc_metrics =  train(model, optimizer,data_loader, epochs, criterion, scheduler )
    plot_metrics(qSc_metrics ) 


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

def plot_metrics(metrics : Dict [str , Dict[str, list]]) : 
    fig, (plot1, plot2) = plt.subplots(nrows=1, ncols=2)
    plot1.plot(metrics['train']['loss'], label='train loss')
    plot1.plot(metrics['validate']['loss'], label='val loss')
    lines, labels = plot1.get_legend_handles_labels()
    plot1.legend(lines, labels, loc='best')

    plot2.plot(metrics['train']['acc'], label='train acc')
    plot2.plot(metrics['validate']['acc'], label='val acc')
    plot2.legend()
    pass 


train_dataset = ds.FashionMNIST(root="./data/FashionMNIST/train" , train = True, download = True, transform=transforms.Compose(transforms.ToTensor(), transforms.Normalize(0.5,)))
validation_dataset = ds.FashionMNIST(root="./data/FashionMNIST/valid" , train = False, download = True , transform=transforms.Compose(transforms.ToTensor(), transforms.Normalize(0.5,)))






# Training system without quantisation hyperparameters training algorithms 
# Train the network with full precision floating point numbers but have quantisation layers observe range of inputs 

# Iteration 1: Initialise quantisation hyperparameters and re train with fake-quantised model (all quantizations to 8-bit ) 
# Iteration 2: Re-Initialise Quantisation hyperparameters to match with HW constraints
# Iteration 3: Train with HW specificities (power of two scales, and whatever)remove nodes and test directly with quantised weights 
# Compare with HW model 