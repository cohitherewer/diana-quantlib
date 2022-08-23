import torch 

real = torch.rand(3) 
imag = torch.rand(3) 

comp = torch.complex(real, imag)
print(comp) 
comp.imag = torch.clip(comp , min=torch.Tensor([0.2]))
print(comp)