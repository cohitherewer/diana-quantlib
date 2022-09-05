
import torch 
from torch import nn 
from torchvision.models import resnet18, ResNet18_Weights
from DianaModules.utils.BaseModules import DianaModule
fp_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)

resnet18_diana = DianaModule(DianaModule.from_trained_model(fp_model, map_to_analog=True) ) 

