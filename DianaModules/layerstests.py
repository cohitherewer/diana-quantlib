import torch
from Digital.DIlayers import DQIdentity

# test identity 

test_tensor  = (torch.rand(3 , 1 , 1  ,3 ) -0.5) * 2 * 5
qrangespec = 'ternary' #{'bitwidth': 8 , 'signed': True} 
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

# test 
