import torch 
from torch import nn
 

test = nn.Sequential(nn.Linear(3,3 ), nn.Linear(3, 1 )) 
for idx ,m in enumerate(test.modules()): 
    print(idx , m) 
    if type(m) == nn.Sequential: 
        print("sequentuel ")
        continue
    rnd = torch.rand(3)
    out = m(rnd)
    print ("type: ", type(m)) 
    print("out:L ", out)