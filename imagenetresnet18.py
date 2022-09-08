
import torch 
from torch import nn 
from torchvision.models import resnet18, ResNet18_Weights
from DianaModules.utils.BaseModules import DianaModule




fp_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)

# Fake - Quantization 
resnet18_diana = DianaModule(DianaModule.from_trained_model(fp_model, map_to_analog=True) ) 
model = resnet18_diana.gmodule

resnet18_diana.initialize_quantization_no_activation()
resnet18_diana.map_scales(HW_Behaviour=True)
# Retraining 

#activation initializaiton 

#activations training if needed 

# HW mapping 

# retrain 

#get final true quantized model 

