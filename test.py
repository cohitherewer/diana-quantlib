filename = "zoo/cifar10/resnet20/FQ-epoch=00-val_acc=0.9196.ckpt" 
from DianaModules.core.Operations import AnalogConv2d
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.models.cifar10.LargeResnet import resnet20
import pytorch_lightning as pl
import torch
from DianaModules.utils.compression.HparamTuner import DistillObjective
from DianaModules.utils.compression.ModelDistiller import QModelDistiller
from DianaModules.utils.compression.QuantStepper import QuantDownStepper 
from DianaModules.utils.serialization.Loader import ModulesLoader  
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as ds
import optuna 


cfg_path = "serialized_models/resnet20.yaml" 
fp_path  = "zoo/cifar10/resnet20/FP_weights.pth"
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
 
#instantiate model and load pre-trained fp 
module = resnet20()  
module.load_state_dict(DianaModule.remove_dict_prefix(torch.load(fp_path, map_location="cpu")["state_dict"])) 
module.eval() 


# edit the from_trainedfp_model function to change the intitial quantization parameters 
model = DianaModule(DianaModule.from_trainedfp_model(module , modules_descriptors=descriptions))

model.attach_train_dataloader(train_dataloader, torch.Tensor([0.03125])) 
model.attach_quantization_dataloader(train_dataloader) 
model.set_quantized(activations=False) 
model.set_optimizer("SGD", lr =0.01)

steps = 6
#stepper = QuantDownStepper(model, steps, initial_quant={"bitwidth": 8, "signed" :True}, target_quant="ternary") 
#stepper.step()  

distiller = QModelDistiller(student =model , teacher=module, learning_rate=0.01 ,momentum=0.4,
           max_epochs=3,weight_decay=5e-5, nesterov=False, lr_scheduler="COSINE" ,gamma= 0.0,optimizer="SGD", seed=42 , warm_up=100) 

trainer = pl.Trainer(max_epochs=1 , accelerator="gpu", devices=[0])
#for _ , mod in model.named_modules(): 
#    if isinstance(mod, AnalogConv2d): 
 #       mod.freeze() 
#for _ , mod in model.named_modules(): 
#    if isinstance(mod ,AnalogConv2d): 
#        mod.enable_scale_opt()

#for _ , mod in model.named_modules(): 
#    if isinstance(mod ,AnalogConv2d): 
#        mod.disable_scale_opt()
#        mod.freeze() 
#trainer = pl.Trainer(max_epochs=1 , gpus=0)
#trainer.fit(distiller, train_dataloader, val_dataloader)
# View weights information before and weights information after (Before and after ternary)
#for i in range(steps+1) : 
#    #if i == 5: 
#    #    trainer.fit(model, train_dataloader, val_dataloader)
#    #    
#    #else: 
#    #if i <4 : 
#    trainer.fit(distiller, train_dataloader, val_dataloader)
#    #else : 
#    #    trainer.fit(model, train_dataloader, val_dataloader)
#        
#    trainer = pl.Trainer(max_epochs=2,accelerator="gpu", devices=[0])
#        
#    if i != steps:
#        stepper.step() 
#    if i==steps-2: 
#        distiller.criterion.alpha = 0.1
#        for _,mod in model.named_modules(): 
#            if isinstance(mod, AnalogConv2d): 
#                mod.thaw() 
#                #mod.enable_scale_only()
#    if i==steps-1: 
#        distiller.criterion.alpha = 0.05
#        distiller.hparams.learning_rate= 0.001
#
#    if i< steps-2: 
#        model.set_quantized(False)
        
trainer = pl.Trainer(max_epochs=24,accelerator="gpu", devices=[0]) 
trainer.fit(model , train_dataloader , val_dataloader)
#def callback(study): 
#    model = study.user_attrs["model"]
#    model.student.set_quantized(activations=False)
#objective = DistillObjective(model, module , train_dataloader,None,base_params= {"max_epochs":4 , "optimizer":"SGD" , "learning_rate":0.1, "momentum": 0.9, "weight_decay":5e-5})
#
## Define the study
#study = optuna.create_study()
## Run the optimization using the Tree-structured Parzen Estimator (TPE) algorithm
#study.optimize(objective, n_trials=100, n_jobs=-1,callbacks=[callback])
#
## Get the best trial
#best_trial = study.best_trial
#
## Print the best hyperparameters
#print("Best hyperparameters: {}".format(best_trial.params))