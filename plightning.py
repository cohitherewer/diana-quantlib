import torchvision.datasets as ds 
import torchvision
import pytorch_lightning as pl 
import torch 
from pathlib import Path
from DianaModules.core.Operations import DIANAConv2d, DIANALinear 
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.models.cifar10.LargeResnet import resnet20
from DianaModules.utils.serialization.Loader import ModulesLoader
from torch.utils.data import DataLoader

from DianaModules.utils.serialization.feature_extraction import AnalogFeatureExtractor

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )
train_dataloader = DataLoader(train_dataset , pin_memory=True , num_workers=28, shuffle=False, batch_size=256)
val_dataloader = DataLoader(test_dataset, pin_memory=True , num_workers=28 , batch_size=256)
train_scale = torch.Tensor([0.03125]) # pow2

output_weights_path = str(Path("zoo/cifar10/workshop/resnet20").absolute() ) 
FP_weights =output_weights_path + "/FP_weights.pth"
model = resnet20() # Floating point 
model.load_state_dict(DianaModule.remove_dict_prefix(torch.load(FP_weights, map_location='cpu')['state_dict']) )
# Fake quantisation layer 
model.eval() 
module_descriptions_pth = "/imec/other/csainfra/nada64/DianaTraining/serialized_models/resnet20.yaml"

#region ModuleLoader
loader = ModulesLoader()
module_descriptions = loader.load(module_descriptions_pth) 
model = DianaModule(DianaModule.from_trainedfp_model(model, module_descriptions) )

model.attach_train_dataloader(train_dataloader, scale = train_scale ) 
model.attach_validation_dataloader(val_dataloader,  scale= train_scale) 

# Initializing quantization 
#trainer = pl.Trainer(accelerator='gpu' , devices=[0])  
#
#trainer.test(model , train_dataloader)
model.start_observing() 
x, _ = train_dataset.__getitem__(0)
x = x.unsqueeze(0) 
_ = model(x)  
model.stop_observing() 
print("initializing quantication")
#model.initialize_quantization(trainer)  #When intiializing the quantization make sure you pass a trainer that has only 1 device. All others can be run on multiple devices. (check the imagenet training for an example)
print("finished quantization")
model.freeze_clipping_bound() 
model.set_optimizer('SGD', lr = 0.1, momentum=0.9 ) 
print("training model")
#trainer.fit(model, train_dataloaders=train_dataloader ,val_dataloaders=val_dataloader ) 
print("finished training model")
print("starting hw mapping")  
model.map_to_hw()
print("finshed hw mapping")  
model.integrize_layers() 
print("finshed layer integrization")   
extractor = AnalogFeatureExtractor(model.gmodule)
extractor.extract(x)
extractor.serialize("test.txt")


#data_folder = Path("backend/cifar10/resnet20")
#model.export_model(str(data_folder.absolute()))
