# Resnet cifar10 classifier 

from collections import OrderedDict
from pathlib import Path
from random import randint
import torch 
from torch import nn
from DianaModules.models.FastResnet import resnet8_mixed 
from DianaModules.models.LargeResnet import resnet20
from DianaModules.utils.BaseModules import DianaModule 
from torchvision import datasets as ds 
import torchvision

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#train_scale = torch.Tensor([1/256]) 
#validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

model = resnet8_mixed()

converted = DianaModule.from_trained_model(model ) 
d_converted = DianaModule(converted)
for idx , module in enumerate(converted.modules()) : 
    print (f'{idx} ---->>>> {module}')
d_converted.attach_validation_dataset(train_dataset )
d_converted.initialize_quantization(5)