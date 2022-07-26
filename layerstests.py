from lib2to3.pytree import convert
import torch
from torch import nn
from DianaModules.core.operations import DIANAConv2d
import DianaModules.core.operations as di
import DianaModules.utils.BaseModules as bm
from DianaModules.utils.DigitalRequant import DigitalRequantizer
from quantlib.editing.editing.editors.retracers import QuantLibRetracer

from quantlib.editing.editing.tests import ILSVRC12, common 
import quantlib.editing.graphs as qg
#from Digital.DIlayers import DQIdentity, DQScaleBias, DQFC



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
#rn18 = ILSVRC12.ResNet.ResNet('ResNet18')
class digital_core_test(nn.Module): # problem with conv of first
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Sequential( nn.Conv2d (3 ,3, 2 ), nn.ReLU())
        self.test_modules = nn.Sequential(nn.Conv2d(3 , 4 , 3, bias=True), nn.ReLU() , nn.Conv2d(4, 7, 3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3 , 7, 5) , nn.ReLU())
    def forward(self, x ): 
        out_1 = self.test_modules(self.conv1(x))
        print(out_1.shape) 
        out_2 = self.conv2(self.conv1(x) ) 
        print(out_2.shape)
        return  out_1+ out_2
        


test_modules = digital_core_test()

from quantlib.editing.graphs.fx import quantlib_symbolic_trace
graph = quantlib_symbolic_trace(root=test_modules)

test_mat = torch.rand(3,3 , 20 ,20 )*255
test_modules(test_mat)

converted_graph = bm.DianaModule.fquantize_model8bit(test_modules) 
converted_graph.start_observing() 

for i in range(10) : 
    test_mat = torch.rand(3,3 , 20 ,20 )*255
    converted_graph(test_mat)
converted_graph.stop_observing()
converted_graph.clip_scales() 
print ("After fake quantization") 
for _ , module in  converted_graph.named_modules(): 
    print (_ , type(module))
#converted_graph.map_scales(HW_Behaviour=True)
# true quant 
converted_graph.true_quantize()

test_mat = (torch.rand(1,3,20,20)*255).floor()

print ("After true quantization") 
for _ , module in enumerate(converted_graph.modules()): 
    print (_ , type(module))
 
    
        
converted_graph.export_model(test_mat)    
#converted_graph.stop_observing()
