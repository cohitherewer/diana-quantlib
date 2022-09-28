from DianaModules.models.cifar10.LargeResnet import resnet20
import torchvision
import torchvision.datasets as ds 
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.serialization.Loader import ModulesLoader
from DianaModules.utils.serialization.Serializer import ModulesSerializer
from DianaModules.core.Operations import DIANAReLU
from DianaModules.utils.verification.SIMDmodelverifier import SIMDVerifier
from pathlib import Path
import torch 
from torch import nn 
import torch.utils.data as ut
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel

output_weights_path = str(Path("zoo/cifar10/workshop/resnet20").absolute() ) 

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )
data_loader = {'train': ut.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True , num_workers=8) , 'validate' : ut.DataLoader(test_dataset, batch_size=128, shuffle=True  ,pin_memory=True, num_workers=8)}
train_scale = torch.Tensor([0.03125]) #found by have train_dataset go thorugh 8-bit quantizer 
device = torch.device('cuda:1') 
FP_weights =output_weights_path + "/FP_weights.pth"
model = resnet20() # Floating point 
model.load_state_dict(DianaModule.remove_data_parallel(torch.load(FP_weights, map_location='cpu')['state_dict']) )

#region float_point_train
###model = nn.DataParallel(model, device_ids=[0 , 1]).cuda() 
###model.to(device)
###current_acc = torch.load(FP_weights)['acc']
###optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001 ) 
###DianaModule.train(model, optimizer ,data_loader=data_loader , epochs=120 , model_save_path=FP_weights , current_acc=current_acc)

#endregion
#region Fake_quantization
module_descriptions_pth = "/imec/other/csainfra/nada64/DianaTraining/serialized_models/resnet20.yaml"
#region ModuleLoader
loader = ModulesLoader()
module_descriptions = loader.load(module_descriptions_pth) 

#endregion
#Base_module definition
_Mixed_model = DianaModule(DianaModule.from_trainedfp_model(model=model , modules_descriptors=module_descriptions))
_Mixed_model.attach_train_dataset(train_dataset, train_scale)
_Mixed_model.attach_validation_dataset(test_dataset,train_scale)
#base_module_end

#region model_serialization(default) 
serializer = ModulesSerializer(_Mixed_model.gmodule)  
serializer.dump(module_descriptions_pth) 
#endregion

#Linear Layers Quantization 
_Mixed_model.gmodule.to(torch.device("cuda:0")) 
_Mixed_model.initialize_quantization()
print("Resnet20 linear layer quantized before training: " , _Mixed_model.evaluate_model()[1] ) 
FQ_weights = output_weights_path + "/FQ_weights.pth" 
test_weights = output_weights_path + "/Ftest_weights.pth" 
#region train FQ
_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule , device_ids=[0,1]).cuda()
#_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(FQ_weights)['state_dict']) )

###print("Current_acc: ", current_acc)
for _, module in _Mixed_model.named_modules(): 
    if isinstance(module, DIANAReLU)  :
        module.freeze() 
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters() , lr=0.1 , momentum=0.4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer= optimizer, mode='max' , factor=0.1, patience=4)
params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=10, model_save_path=test_weights , scheduler = scheduler) # training with scale of layer before relu clipped to 2 
for _, module in _Mixed_model.named_modules(): 
    if isinstance(module, DIANAReLU)  :
        module.thaw() 
print("FINISHED TRAINING WITHOUT ACT")
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters() , lr=0.001 , momentum=0.4) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer= optimizer, mode='max' , factor=0.1, patience=4)
params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=10, model_save_path=test_weights , scheduler = scheduler) # training with scale of layer before relu clipped to 2 

#endregion 
_Mixed_model.gmodule.to(device)
print("Resnet20 linear layer quantized after training: " , _Mixed_model.evaluate_model()[1] )  
_Mixed_model.initialize_quantization_activations()
FQ_weights_act = output_weights_path + "/FQ_weights_act.pth" 
print("Resnet20 acts quantized before training: " , _Mixed_model.evaluate_model()[1] ) 
_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(FQ_weights_act)['state_dict']) )
#_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule , device_ids=[0,1]).cuda() 
_Mixed_model.gmodule.to(device) 
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters() , lr=0.001, momentum = 0.3) 
current_acc = torch.load(FQ_weights_act)['acc']
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer= optimizer, mode='max' , factor=0.1, patience=4)
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=120, model_save_path=FQ_weights_act, scheduler = scheduler, current_acc=current_acc ) # training with scale of layer before relu clipped to 2 

print("Resnet20 acts quantized after training: " , _Mixed_model.evaluate_model()[1] ) 
#endregion
_Mixed_model.gmodule.to(torch.device('cpu'))
_Mixed_model.map_to_hw()

HWmapped_weights = output_weights_path + "/HWmapped_weights.pth" 
_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(HWmapped_weights)['state_dict']))
#_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule , device_ids=[0,1]).cuda() 
#_Mixed_model.gmodule.to(device) 
#print("HW mapping before training" , _Mixed_model.evaluate_model()[1] ) 
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters() , lr=0.01, momentum = 0.9) 
current_acc = torch.load(HWmapped_weights)['acc']
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer= optimizer, mode='max' , factor=0.1)
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=120, model_save_path=HWmapped_weights, scheduler = scheduler, current_acc=current_acc ) # training with scale of layer before relu clipped to 2 
#_Mixed_model.gmodule.to(device) 
#print("HW mapping after training" , _Mixed_model.evaluate_model()[1] ) 


_Mixed_model.gmodule.to(torch.device('cpu')) 
_Mixed_model.integrize_layers()
_Mixed_model.gmodule.to(device) 
print("integrized solution acc: " , _Mixed_model.evaluate_model()[1] )  

data_folder = Path("backend/cifar10/resnet20")

_Mixed_model.gmodule.to('cpu')
_Mixed_model.export_model(str(data_folder.absolute()))



