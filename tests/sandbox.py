import torch 
from torch import nn

# flattening 
rt = torch.rand(2, 3, 3,3)

rt= rt.flatten(start_dim=1) 
print(rt.size())