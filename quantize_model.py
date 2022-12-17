from DianaModules.utils.BaseModules import DianaModule
from DianaModules.models.cifar10.LargeResnet import resnet20
import pytorch_lightning as pl
import torch 
from DianaModules.utils.serialization.Loader import ModulesLoader  
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as ds
cfg_path = "serialized_models/resnet20.yaml" 
fp_path  = "zoo/cifar10/resnet20/FP_weights.pth"
out_path = "zoo/cifar10/resnet20/quantized_8b.pth"
#define dataset 
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

#define dataloader 
train_dataloader = DataLoader(train_dataset , pin_memory=True , num_workers=28, shuffle=False, batch_size=256)
val_dataloader = DataLoader(test_dataset, pin_memory=True , num_workers=28 , batch_size=256)
#Module laoder 
loader = ModulesLoader()
descriptions = loader.load(cfg_path)
#quantize model 
trainer = pl.Trainer(accelerator="gpu", strategy = "dp" ,devices=-1)   
#instantiate model and load pre-trained fp 
module = resnet20()  
module.load_state_dict(DianaModule.remove_dict_prefix(torch.load(fp_path, map_location="cpu")["state_dict"])) 
module.eval() 

#trainer.validate(DianaModule(module) ,val_dataloader)

# edit the from_trainedfp_model function to change the intitial quantization parameters 
model = DianaModule(DianaModule.from_trainedfp_model(module , modules_descriptors=descriptions))
model.initialize_quantization_layers(trainer, train_dataloader)

torch.save ({
                       'state_dict': model.gmodule.state_dict(),
                    } , out_path)

# python init trainer: 
# python trainer.py --lr 0.01 --weight_decay 5e-4 --batch_size 256 --num_workers 8 --num_epochs 20 --scale 0.03125 --quant_steps 0 --config_pth ./serialized_models/resnet20.yaml --stage fq --fp_pth ./zoo/cifar10/resnet20/ResNet_FPweights.pth --quantized_pth ./zoo/cifar10/resenet20/quantized.pth --checkpoint_dir ./zoo/cifar10/resnet20
# using distiller 
# python trainer.py --lr 0.001 --weight_decay 5e-5 --batch_size 256 --num_workers 8 --num_epochs 20 --scale 0.03125 --quant_steps 2 --config_pth ./serialized_models/resnet20.yaml --stage fq --fp_pth ./zoo/cifar10/resnet20/FP_weights.pth --quantized_pth ./zoo/cifar10/resnet20/quantized_8b.pth --checkpoint_dir ./zoo/cifar10/resnet20