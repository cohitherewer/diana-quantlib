# Resnet cifar10 classifier 

from pathlib import Path
import torch 
from torch import nn 
import DianaModules.models.resnet as resnet
from DianaModules.utils.BaseModules import DianaModule 
from torchvision import datasets as ds 
import torchvision

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

model = getattr(resnet, f'cifar10_resnet20')(pretrained=True)

weight_file = Path("trained_models/resnet20/CifarResNet_FPweights.pth")
#model.load_state_dict(torch.load(str(weight_file.absolute())))
#model.eval()
#converted = DianaModule.from_trained_fp_model(model ) 
#for idx , module in enumerate(converted.modules()) : 
#    print (f'{idx} ---->>>> {module}')
##TRAINING

converted = DianaModule(model)
converted.attach_train_dataset(train_dataset , train_scale) 
converted.attach_validation_dataset(validation_dataset, train_scale)


data_folder = Path("trained_models/resnet20")
converted.QA_iterative_train(epochs=20, batch_size=128, train_FP_model=False ) 

