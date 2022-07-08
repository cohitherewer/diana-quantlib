import torch
from torch import nn

def quantized_aware_train(model: nn.Module): 

    pass 


# Training system without quantisation hyperparameters training algorithms 
# Step1: 1. Train the network with full precision floating point numbers but have quantisation layers observe range of inputs 
# step2 : initialise quantisation hyperparameters and re train with quantised model 
# step3 : remove nodes and test directly with quantised weights 
