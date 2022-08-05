# Resnet cifar10 classifier 

import torch 
from torch import nn 
from DianaModules.models.resnet import ResNet18 
from DianaModules.utils.BaseModules import DianaModule 
from torchvision import datasets as ds 
import torchvision

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True)  

model = ResNet18()

#converted = DianaModule.from_trained_fp_model(model ) 
#for idx , module in enumerate(converted.modules()) : 
#    print (f'{idx} ---->>>> {module}')
##TRAINING
converted = DianaModule(model ) 
converted.attach_train_dataset(train_dataset , train_scale) 
converted.attach_validation_dataset(validation_dataset)

converted.QA_iterative_train(epochs=3 ) 

