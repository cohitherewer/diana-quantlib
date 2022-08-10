import torch 
import torch.nn as nn
import torch.nn.functional as F

import torch 
from torch import nn 
from DianaModules.models.resnet import ResNet18 
from DianaModules.utils.BaseModules import DianaModule 
from torchvision import datasets as ds 
import torchvision

#train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True, transform=torchvision.transforms.ToTensor())  


if torch.cuda.is_available(): 
  print("Let's use", torch.cuda.device_count(), "GPUs!")


