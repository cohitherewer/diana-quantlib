from argparse import ArgumentParser
from pathlib import Path
import torchvision
import torch.nn as nn 
import torchvision.datasets as ds 
from torch.utils.data import DataLoader
import torch 
from DianaModules.models.cifar10.MobileNet import MobileNetV2

from DianaModules.models.imagenet.Dataset import ImagenetTrainDataset
from DianaModules.models.imagenet.Resnet import resnet18_imgnet
from DianaModules.utils.BaseModules import DianaModule 
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
#train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
#            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_dataset = ImagenetTrainDataset() 
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )
train_dataloader = DataLoader(train_dataset , pin_memory=True , num_workers=8 , shuffle=True, batch_size=512)
val_dataloader = DataLoader(test_dataset)
train_scale = torch.Tensor([0.03125]) # pow2
mod = resnet18_imgnet() # CIFAR 10 

model = DianaModule(DianaModule.from_trainedfp_model(mod))
model.attach_train_dataloader(train_dataloader , scale=train_scale)
model.start_observing() 
x, _ = train_dataset.__getitem__(0)
x = x.unsqueeze(0) 
_ = model(x)  
model.stop_observing() 
print("FQ converted")
model.map_to_hw() 
print("HW mapped ") 
for _ , mod in model.gmodule.named_modules(): 
    if isinstance(mod , _QModule): 
        print(mod)
model.integrize_layers() 
print("layer integrize ")
for node in model.gmodule.graph.nodes : 
    try: 
        if (isinstance(model.gmodule.get_submodule(node.target), _QModule)): 
            users = [u for u in node.users]
            pred = [p for p in node.all_input_nodes]

            print(f"Qmodule Node: {node} has the users: ", *users, " and pred: ", *pred) 
        elif (isinstance(model.gmodule.get_submodule(node.target), nn.BatchNorm2d)): 
            users = [u for u in node.users]
            pred = [p for p in node.all_input_nodes]

            print(f"BN Node: {node} has the users: ", *users, " and pred: ", *pred) 
        pass 
    except : 
        pass 


data_folder = Path("backend/cifar10/resnet18")

model.export_model(str(data_folder.absolute()))




