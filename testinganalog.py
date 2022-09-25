

# Resnet cifar10 classifier 

from turtle import forward
import torch 
from torch import nn

from DianaModules.core.Operations import AnalogAccumulator, AnalogConv2d
from DianaModules.utils.BaseModules import DianaModule 
import torchvision 
import torchvision.datasets as ds 

# Testing equality 
#$$class aconv(nn.Module) : 
#$$    def __init__(self , in_channels , out_channels , kernel_size  ,stride ,padding  ) -> None:
#$$        super().__init__()
#$$        self.conv1 = AnalogConv2d('ternary' , 'per-array', 'meanstd' , in_channels , out_channels , kernel_size , stride , padding)
#$$        self.acc = AnalogAccumulator()
#$$    def forward(self, x : torch.Tensor): 
#$$        return self.acc(self.conv1(x)) 
#$$
#$$test_input = torch.rand(128 ,128 ,24 ,24 )
#$$regular_conv = nn.Conv2d(128 , 256 ,2,1,1, bias=False)
#$$analog_conv = aconv(128 , 256 , 2 ,1 ,1)
#$$
#$$analog_conv.conv1.weight.data =  regular_conv.weight.data.clone()
#$$
#$$out_reg_conv = regular_conv(test_input) 
#$$out_ana_conv = analog_conv(test_input) 
#$$print( out_ana_conv.shape)
#$$print( out_reg_conv.shape)
#$$print((out_ana_conv -out_reg_conv).max()) # difference of e-7 basically equal

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

# Testing conversion
class model (nn.Module)  : 
    def __init__(self) -> None:
        super().__init__()
        self.sq = nn.Sequential(nn.Conv2d(3 , 5, 2,1,1), nn.BatchNorm2d(5)) 
    def forward(self , x: torch.Tensor): 
        return self.sq(x) 
m = model()
model_converted = DianaModule(DianaModule.from_trainedfp_model(m , map_to_analog=True) )
print("FAKE QUANTIAATIONN")
print("FAKE QUANTIAATIONN")
print(model_converted.gmodule)
print("FAKE QUANTIAATIONN")
print("FAKE QUANTIAATIONN")
model_converted.attach_train_dataset(train_dataset ,train_scale)
model_converted.attach_validation_dataset(validation_dataset,train_scale)
model_converted.initialize_quantization(5) 
model_converted.true_quantize() 
print("TRUE QUANTIAATIONN")
print("TRUE QUANTIAATIONN")
print(model_converted.gmodule)
print("TRUE QUANTIAATIONN")
print("TRUE QUANTIAATIONN")
#d_converted = DianaModule(converted)
#for idx , module in enumerate(converted.modules()) : 
#    print (f'{idx} ---->>>> {module}')
#d_converted.attach_validation_dataset(train_dataset )
#d_converted.initialize_quantization(5)