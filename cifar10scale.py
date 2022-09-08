import torchvision.datasets as ds  
import torchvision
import torch

from DianaModules.core.Operations import DIANAIdentity 

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#train_scale = torch.Tensor([2**-8]) 
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

a = DIANAIdentity({'bitwidth': 8 , 'signed': True} , 'per-array', 'minmax' )
a .start_observing() 
a.to(torch.device('cuda:0')) 

for i in range(len(test_dataset )):     
    x, _ = test_dataset.__getitem__(i)
    _ = a(x) 

a.stop_observing()

print(a.scale) 