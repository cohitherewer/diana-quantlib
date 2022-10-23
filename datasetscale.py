from email.mime import image
import torchvision.datasets as ds  
import torchvision
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
import time 
from DianaModules.core.Operations import DIANAIdentity
from DianaModules.models.imagenet.Dataset import ImagenetTrainDataset 

# for imagenet 
sample_every_ith = 2
class CustomDataloader(DataLoader): 
    def _get_iterator(self):
        #return super()._get_iterator()
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return multiprocessingiter(self)
        
class multiprocessingiter(_MultiProcessingDataLoaderIter) :
    pass
    def _next_index(self):
        for i in range (sample_every_ith-1) : 
            next(self._sampler_iter)
        return super()._next_index() 

        
# for imagenet end 
#train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
#            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
device = torch.device('cuda:1') 
imagenet_train_dataset = ImagenetTrainDataset() 
batch_size = 256
loader = CustomDataloader(imagenet_train_dataset ,batch_size= batch_size, num_workers=8 ,pin_memory=True , shuffle=True  )

a = DIANAIdentity({'bitwidth': 8 , 'signed': True} , 'per-array', 'minmax' )
b = DIANAIdentity({'bitwidth': 8 , 'signed': True} , 'per-array', 'meanstd' )

a .start_observing() 
b.start_observing()

model = nn.Sequential(a , b)
a.to(device) 
b.to(device) 
torch.cuda.synchronize(device)

t0 = time.time()
counter = 0 
it = 0 
for i,data in enumerate(loader):  
    if (it == 3) : 
        it = 0 
    else: 
        it += 1
        continue 
    # time it               
    x , y = data[0].to(device), data[1].to(device)
    _ = a(x) 
    _ = b(x) 
    counter +=1 
    print(f"finished loop: {counter} ")
torch.cuda.synchronize(device)
t1 = time.time()
print(f"total time{t1-t0}") 

a.stop_observing()
b.stop_observing()

print("(minmax) scale" , a.scale) 
print("max float (minmax): " , a.max_float)
print("min float (minmax): " , a.min_float)

print("(meanstd) scale" , b.scale) 
print("max float (meanstd): " , b.max_float)
print("min float (meanstd): " , b.min_float)


# CIFAR 10: 
#(minmax) scale tensor([0.0217], device='cuda:0')
#max float (minmax):  tensor([2.7537])
#min float (minmax):  tensor([-2.4291])
#(meanstd) scale tensor([0.0354], device='cuda:0')
#max float (meanstd):  tensor([3.8810])
#min float (meanstd):  tensor([-4.5335])

#image net: (sampling every 4th batch )
#finished loop: 312 
#total time199.08788013458252
#(minmax) scale tensor([0.0208], device='cuda:1')
#max float (minmax):  tensor([2.6400], device='cuda:1')
#min float (minmax):  tensor([-2.1179], device='cuda:1')
#(meanstd) scale tensor([0.0281], device='cuda:1')
#max float (meanstd):  tensor([3.5674], device='cuda:1')
#min float (meanstd):  tensor([-3.6013], device='cuda:1')

#image net: (sampling every 2nd batch ) 
#finished loop: 625 
#total time393.5815579891205
#(minmax) scale tensor([0.0208], device='cuda:1')
#max float (minmax):  tensor([2.6400], device='cuda:1')
#min float (minmax):  tensor([-2.1179], device='cuda:1')
#(meanstd) scale tensor([0.0281], device='cuda:1')
#max float (meanstd):  tensor([3.5702], device='cuda:1')
#min float (meanstd):  tensor([-3.6032], device='cuda:1')