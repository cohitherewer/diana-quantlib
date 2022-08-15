# FOR Architecture go to : https://github.com/davidcpage/cifar10-fast
import torch 
from torch import nn 
# Model definitions 
PREP_LAYER_CHANNELS = 32 
LAYER1_CHANNELS = 64 
LAYER2_CHANNELS = 128
LAYER3_CHANNELS = 256
POOL_SIZE = 2
LAYER1_RES_KWARGS = {'in_channels': LAYER1_CHANNELS, 'out_channels': LAYER1_CHANNELS , 'kernel_size' : 3 , 'padding' : 1, 'bias':False }
LAYER3_RES_KWARGS = {'in_channels': LAYER3_CHANNELS, 'out_channels': LAYER3_CHANNELS , 'kernel_size' : 3 , 'padding' : 1 , 'bias': False}
class resnet8(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.layer_0 = nn.Sequential(nn.Conv2d(3 , PREP_LAYER_CHANNELS , 3, padding=1) , nn.BatchNorm2d(PREP_LAYER_CHANNELS)  , nn.ReLU() ) 
        
        
        self.layer_1_prep =  nn.Sequential(nn.Conv2d(in_channels=PREP_LAYER_CHANNELS , out_channels=LAYER1_CHANNELS , kernel_size=3 , padding=1 , bias =False)  , nn.BatchNorm2d(LAYER1_CHANNELS) , nn.ReLU(), nn.MaxPool2d(POOL_SIZE))
        self.layer_1_res_add = nn.Sequential(nn.Conv2d(**LAYER1_RES_KWARGS), nn.BatchNorm2d(LAYER1_CHANNELS) , nn.ReLU() , nn.Conv2d(**LAYER1_RES_KWARGS), nn.BatchNorm2d(LAYER1_CHANNELS) , nn.ReLU())

        
        self.layer_2_prep =  nn.Sequential(nn.Conv2d(in_channels=LAYER1_CHANNELS,out_channels=LAYER2_CHANNELS , kernel_size=3 , padding=1 )  , nn.BatchNorm2d(LAYER2_CHANNELS) , nn.ReLU(), nn.MaxPool2d(POOL_SIZE))
        
 
        
        self.layer_3_prep =  nn.Sequential(nn.Conv2d(in_channels=LAYER2_CHANNELS , out_channels=LAYER3_CHANNELS, kernel_size=3 , padding=1 )  , nn.BatchNorm2d(LAYER3_CHANNELS) , nn.ReLU(), nn.MaxPool2d(POOL_SIZE))
        self.layer_3_res_add = nn.Sequential(nn.Conv2d(**LAYER3_RES_KWARGS), nn.BatchNorm2d(LAYER3_CHANNELS) , nn.ReLU() , nn.Conv2d(**LAYER3_RES_KWARGS), nn.BatchNorm2d(LAYER3_CHANNELS) , nn.ReLU())
        
        self.head = nn.ModuleList([nn.MaxPool2d(4), nn.Linear(LAYER3_CHANNELS, 10 , bias =False)])
    def forward(self , x : torch.Tensor) : 
        o1 = self.layer_0  (x) 
        prep1 = self.layer_1_prep(o1) 
        add1 = prep1 + self.layer_1_res_add(prep1) 

        o2 = self.layer_2_prep(add1)

        prep3 = self.layer_3_prep(o2) 
        add3 = prep3 + self.layer_3_res_add(prep3)  
        pool = self.head[0](add3) 

        pool = pool.view(pool.size(0), -1) 
        out = self.head[1](pool) 
        return out 
class resnet8_mixed(resnet8): 
    def __init__(self) : 
        super().__init__() 
        self.layer_0 = nn.Sequential (nn.Conv2d(3 , PREP_LAYER_CHANNELS , 3 , padding=1) , nn.ReLU()) # 8 bits 