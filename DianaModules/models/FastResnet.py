# FOR Architecture go to : https://github.com/davidcpage/cifar10-fast
import torch 
from torch import nn 
# Model definitions 
PREP_LAYER_CHANNELS = 64 
class resnet8(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.layer_0 = nn.Conv2d(3 , 64 , 3, padding=1) 
        self.layer_1 = nn
        self.layer_1_res_add = 