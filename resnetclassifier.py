# Resnet cifar10 classifier 

from collections import OrderedDict
from pathlib import Path
import torch 
from torch import nn 
from DianaModules.models.resnet import resnet20
from DianaModules.utils.BaseModules import DianaModule 
from torchvision import datasets as ds 
import torchvision
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

model = resnet20()
weight_file = Path("trained_models/resnet20/ResNet_FPweights.pth")
model.load_state_dict(torch.load(str(weight_file.absolute())))
model.eval()

#converted = DianaModule.from_trained_fp_model(model ) 
#for idx , module in enumerate(converted.modules()) : 
#    print (f'{idx} ---->>>> {module}')
##TRAINING

converted = DianaModule(model)
converted.attach_train_dataset(train_dataset , train_scale) 
converted.attach_validation_dataset(validation_dataset, train_scale)


data_folder = Path("trained_models/resnet20")
converted.QA_iterative_train(epochs=3, batch_size=128, train_FP_model=False , output_weights_path=str(data_folder.absolute()) ) 

