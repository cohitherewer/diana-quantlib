import torch 
from torch import nn

#test = nn.Sequential(nn.Linear(3,3 ), nn.Linear(3, 1 )) 
#for idx ,m in enumerate(test.modules()): 
#    print(idx , m) 
#    if type(m) == nn.Sequential: 
#        print("sequentuel ")
#        continue
#    rnd = torch.rand(3)
#    out = m(rnd)
#    print ("type: ", type(m)) 
#    print("out:L ", out)
# target output size of 5x7
#m = nn.AdaptiveAvgPool2d((5,7))
#input = torch.randn(1, 64, 8, 9)
#output = m(input)
#print(output.size())
## target output size of 7x7 (square)
#m = nn.AdaptiveAvgPool2d(7)
#input = torch.randn(1, 64, 10, 9)
#output = m(input)
## target output size of 10x7
#m = nn.AdaptiveAvgPool2d((None, 7))
#input = torch.randn(1, 64, 10, 9)
#output = m(input)

b = torch.Tensor([[1], [1], [1]])
te = torch.rand(b.shape)
print(type(b.shape)) 