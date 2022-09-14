
import torch 
from torch import nn 
import time 
from torchvision.models import resnet18, ResNet18_Weights
from DianaModules.models.imagenet.Dataset import ImagenetTrainDataset, ImagenetValidationDataset
from DianaModules.utils.BaseModules import DianaModule
from torch.utils.data import DataLoader

train_dataset = ImagenetTrainDataset()
validation_dataset = ImagenetValidationDataset()

batch_size = 256 
epochs = 90 
lr = 0.1 
momentum = 0.9 
num_workers = 8  
weight_decay = 1e-4 
device = torch.device('cuda:1')
fp_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
data_loader = {'train': DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4) , 'validate' : DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True  ,pin_memory=True, num_workers=4)}
# Fake - Quantization 
resnet18_diana = DianaModule(DianaModule.from_trained_model(fp_model, map_to_analog=False) ) 
resnet18_diana.attach_train_dataset(train_dataset)
resnet18_diana.attach_validation_dataset(validation_dataset)
resnet18_diana.gmodule.to(device) 
torch.cuda.synchronize(device)
t0 = time.time()
print("FP model acc  ", resnet18_diana.evaluate_model()[1])
torch.cuda.synchronize(device)
t1= time.time()
print( f"it took {t1-t0} second to evaluate the fp model") 

print("initializing linear layers quantization")
torch.cuda.synchronize(device)
t0 = time.time()
resnet18_diana.initialize_quantization_no_activation(10000)
print("finished initializing linear layers quantization")
torch.cuda.synchronize(device)
t1= time.time()
print( f"it took {t1-t0} second to quantized linear layers")
print("FQ model with linear layers quantized acc  ", resnet18_diana.evaluate_model()[1])
print("initializing activations quantization")
resnet18_diana.initialize_quantization_activations(10000)
print("finished initializing activations quantization")
print("FQ model with activations quantized acc  ", resnet18_diana.evaluate_model()[1])

# Retraining 

#activation initializaiton 

#activations training if needed 

# HW mapping 

# retrain 

#get final true quantized model 
#batch size 256 , lr = 0.1 , momentum = 0.9 ,weight_decay 1e-4
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# imagenet scaling 8-bits 