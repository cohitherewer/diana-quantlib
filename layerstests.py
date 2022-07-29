from lib2to3.pytree import convert
from turtle import down, forward
import torch
from torch import nn
from DianaModules.core.operations import DIANAConv2d, DIANAIdentity
import DianaModules.core.operations as di
import DianaModules.utils.BaseModules as bm
from DianaModules.utils.DigitalRequant import DigitalRequantizer
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.tests import ILSVRC12


import quantlib.editing.graphs as qg
from torchvision import datasets as ds
import torchvision.transforms
from DianaModules.models.resnet import ResNet18
import torch.nn.functional as F

# test identity 

#test_tensor  = (torch.rand(3 , 1 , 1  ,3 ) -0.5) * 2 * 5
#qrangespec = 'ternary' 
#qgranularityspec = 'per-array' 
#qhparamsinitstrategyspec  ='minmax' # clamp between 1 and -1 



#test_act = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec ,qhparamsinitstrategyspec=qhparamsinitstrategyspec)  
#test_act.start_observing() 
#output = test_act(test_tensor) 
#test_act.stop_observing()
#test_tensor  = (torch.rand(3 , 1 , 1  ,3 )  - 0.5 )*2 

#output = test_act(test_tensor) 
#print(test_act.scale) 
#print("BEFORE IDENTITY" , test_tensor)

#print("################")
#output = test_act(test_tensor) 
#print(output)

# BN  
#qrangespec =  {'bitwidth': 8 , 'signed': False} 
#qgranularityspec = 'per-array' 
#qhparamsinitstrategyspec  = 'const'

#test_bn = DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec ,qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=3 ) 
#test_tensor = torch.rand(6 , 3, 28, 28 )
#test_bn.start_observing()
#for i in range(20): 
    #test_tensor = torch.rand(6 , 3, 28, 28 )
    #test_bn(test_tensor) 
#test_bn.stop_observing() 

#test_tensor = torch.rand(1,3,2,2) 
#output = test_bn(test_tensor)



 # test quanttensor matmul 



# testing quant tensor operations 

#x = QuantTensor(torch.rand( 3 ,1 )) 
#y = QuantTensor(torch.rand(3, 1))
#print (type(x) , type(y) ) 
#print(torch.mul(x , y) )
#print("##################")
#print (torch.matmul(x, y))
 
 # Testing creation of fake quantized models of diana module from FP models 
class   DBlock(nn.Module): 

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
           
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
     

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = F.relu(self.conv1(input), inplace = True)
        input = F.relu(self.conv2(input), inplace=True)
        input = input + shortcut
        return F.relu(input, inplace =True)
class dcore_network(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.layer_0 =  nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, stride=1, padding=1 ) ,nn.ReLU())
        self.resblock_layers = nn.Sequential(
            DBlock(64, 64),
            DBlock(64, 64)
        ,

            DBlock(64, 128, downsample=True),
            DBlock(128, 128)
        ,

            DBlock(128, 256, downsample=True),
            DBlock(256, 256)
        , 
            DBlock(256, 512, downsample=True),
            DBlock(512, 512)
        )
    def forward(self, x: torch.Tensor) : 
        out1 = self.layer_0(x)
        return self.resblock_layers(out1) 
         

#test_model = ResNet18()
test_model = dcore_network() 


dataset = ds.CIFAR10('./data/cifar10', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
scale = 1
test_indices = list(range(0, 100))


converted_graph = bm.DianaModule.fquantize_model8bit(test_model) 


converted_graph.start_observing() 

for i in  test_indices :
    x = dataset.__getitem__(i)[0].unsqueeze(0) 

    _ = converted_graph(x) 


converted_graph.stop_observing() 
converted_graph.clip_scales() 
print ("After fake quantization") 
for _ , module in  converted_graph.named_modules(): 
    print (_ , type(module))
#converted_graph.map_scales(HW_Behaviour=True)
# true quant 
converted_graph.attach_train_dataset(dataset , torch.Tensor([scale]))
converted_graph.true_quantize([ ILSVRC12.ResNet.RNHeadRewriter()])

print ("After true quantization") 

        
converted_graph.export_model()    
#converted_graph.stop_observing()

