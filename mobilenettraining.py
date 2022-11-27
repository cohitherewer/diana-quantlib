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
from DianaModules.utils.serialization.Loader import ModulesLoader
from DianaModules.utils.serialization.Serializer import ModulesSerializer 
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#train_dataset = ImagenetTrainDataset() 
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )
train_dataloader = DataLoader(train_dataset , pin_memory=True , num_workers=8 , shuffle=True, batch_size=512)
val_dataloader = DataLoader(test_dataset)
train_scale = torch.Tensor([0.03125]) # pow2
loader = ModulesLoader()
module_descriptions_pth = "/imec/other/csainfra/nada64/DianaTraining/serialized_models/Mobilenetv2.yaml"
module_descriptions = loader.load(module_descriptions_pth) 
mod = MobileNetV2() # CIFAR 10 

model = DianaModule(DianaModule.from_trainedfp_model(mod , modules_descriptors=module_descriptions_pth))

#serializer = ModulesSerializer(model.gmodule)  
#serializer.dump(module_descriptions_pth) 
model.attach_train_dataloader(train_dataloader , scale=train_scale)
model.start_observing() 
x, _ = train_dataset.__getitem__(0)
x = x.unsqueeze(0) 
_ = model(x)  
model.stop_observing() 
print("FQ converted")
for node in model.gmodule.graph.nodes : 
    try: 
        if (isinstance(model.gmodule.get_submodule(node.target), nn.BatchNorm2d)): 
            users = [u for u in node.users]
            pred = [p for p in node.all_input_nodes]

            print(f"BN Node: {node} has the users: ", *users, " and pred: ", *pred) 
        pass 
    except : 
        pass 
model.map_to_hw() 
print("HW mapped ") 
for node in model.gmodule.graph.nodes : 
    try: 
        if (isinstance(model.gmodule.get_submodule(node.target), nn.BatchNorm2d)): 
            users = [u for u in node.users]
            pred = [p for p in node.all_input_nodes]

            print(f"BN Node: {node} has the users: ", *users, " and pred: ", *pred) 
        pass 
    except : 
        pass 
model.integrize_layers() 
print("layer integrize ")



#data_folder = Path("backend/cifar10/resnet18")

#model.export_model(str(data_folder.absolute()))




