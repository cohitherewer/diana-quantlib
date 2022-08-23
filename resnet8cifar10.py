from DianaModules.models.FastResnet import resnet8  , resnet8_mixed
from DianaModules.utils.BaseModules import DianaModule
from pathlib import Path 
import torchvision 
import torchvision.datasets as ds  
import torch 
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

#model = resnet8() 
model = resnet8_mixed()

diana_model = DianaModule(model)  

diana_model.attach_train_dataset(train_dataset , train_scale) 
diana_model.attach_validation_dataset(validation_dataset, train_scale)

data_folder = Path("zoo/cifar10/resnet8_mixed_new")
diana_model.QA_iterative_train(epochs=1, batch_size=16, train_FP_model=False, output_weights_path=str(data_folder.absolute())) 

