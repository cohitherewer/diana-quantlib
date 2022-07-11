import torch
from Digital.DIlayers import DQIdentity, DQScaleBias, DQFC

# test identity 

test_tensor  = (torch.rand(3 , 1 , 1  ,3 ) -0.5) * 2 * 5
qrangespec = 'ternary' 
qgranularityspec = 'per-array' 
qhparamsinitstrategyspec  ='minmax' # clamp between 1 and -1 



test_act = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec ,qhparamsinitstrategyspec=qhparamsinitstrategyspec)  
test_act.start_observing() 
output = test_act(test_tensor) 
test_act.stop_observing()
test_tensor  = (torch.rand(3 , 1 , 1  ,3 )  - 0.5 )*2 

output = test_act(test_tensor) 
print(test_act.scale) 
print("BEFORE IDENTITY" , test_tensor)

print("################")
output = test_act(test_tensor) 
print(output)

# BN  
qrangespec =  {'bitwidth': 8 , 'signed': False} 
qgranularityspec = 'per-array' 
qhparamsinitstrategyspec  = 'const'

test_bn = DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec ,qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=3 ) 
test_tensor = torch.rand(6 , 3, 28, 28 )
test_bn.start_observing()
for i in range(20): 
    test_tensor = torch.rand(6 , 3, 28, 28 )
    test_bn(test_tensor) 
test_bn.stop_observing() 

test_tensor = torch.rand(1,3,2,2) 
output = test_bn(test_tensor)
print (test_tensor)
print("########################################")
print(output, test_bn.get_weight_scale(), test_bn.get_bias_scale())

