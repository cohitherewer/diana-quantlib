
from pathlib import Path
import torch 
from torch import nn 
import time 
from DianaModules.models.imagenet.Dataset import ImagenetTrainDataset, ImagenetValidationDataset
from DianaModules.models.imagenet.Resnet import resnet18_imgnet
from DianaModules.utils.BaseModules import DianaModule
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter
from DianaModules.utils.serialization.Loader import ModulesLoader

from DianaModules.utils.serialization.Serializer import ModulesSerializer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping , ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
# Args passer  



# Imagenet datalaoder setup
train_dataset = ImagenetTrainDataset()
validation_dataset = ImagenetValidationDataset()

# for imagenet 
sample_every_ith = 2
class CustomDataloader(DataLoader): 
    def _get_iterator(self):
        #return super()._get_iterator()
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return multiprocessingiter(self)        
class multiprocessingiter(_MultiProcessingDataLoaderIter) :
    pass
    def _next_index(self):
        for i in range (sample_every_ith-1) : 
            next(self._sampler_iter)
        return super()._next_index() 

import os 

workers = min(4,int(os.cpu_count() / 2)) 
batch_size = 256

quant_dataloader  = CustomDataloader(train_dataset ,batch_size= batch_size, num_workers=workers ,pin_memory=True , shuffle=True  )
train_dataloader= DataLoader(train_dataset,batch_size= batch_size, num_workers=workers ,pin_memory=True , shuffle=True ) 
validation_dataloader = DataLoader(validation_dataset,batch_size= batch_size, num_workers=workers ,pin_memory=True ) 

#imagenet_scale  = torch.Tensor([0.0208])
imagenet_scale = torch.Tensor([0.03125 ]) #closest pow2 (floor)


fp_model = resnet18_imgnet()
fp_model.eval() 
module_descriptions_pth = "/imec/other/csainfra/nada64/DianaTraining/serialized_models/resnet18.yaml"
#region ModuleLoader
loader = ModulesLoader()
module_descriptions = loader.load(module_descriptions_pth) 

#Training 
#Prameter setup 
max_epochs = 90 
lr = 0.01 
momentum = 0.9 
weight_decay = 5e-4 

#quant initializatin
print("trainer init")

#print("validating fp")
#trainer.validate(resnet18_diana , validation_dataloader)
print("training floating point model ")
fp = DianaModule(fp_model)
fp.set_optimizer('SGD', lr =0.1 , momentum=0.9, weight_decay=weight_decay)
checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath='zoo/imagenet/resnet18/FP/',filename='imgnet-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
callbacks = [checkpoint_callback  ]
max_epochs = 120
#trainer = pl.Trainer(accelerator = "gpu",strategy="ddp",  devices = -1 , max_epochs =max_epochs , callbacks=callbacks,logger=CSVLogger(save_dir="logs/FP"))
#trainer.fit(fp ,train_dataloader , validation_dataloader)
fp.load_state_dict(torch.load("zoo/imagenet/resnet18/FP/imgnet-epoch=100-val_acc=0.6550.ckpt", map_location="cpu")["state_dict"])
print("testing fp")
#trainer.validate(fp ,validation_dataloader)
print("STARTING FQ")
resnet18_diana = DianaModule(DianaModule.from_trainedfp_model(fp.gmodule, modules_descriptors=module_descriptions) ) 
print("FINISHED FQ")
resnet18_diana.attach_quantization_dataloader(quant_dataloader) 
resnet18_diana.attach_train_dataloader(train_dataloader)
resnet18_diana.attach_validation_dataloader(validation_dataloader) 
serializer = ModulesSerializer(resnet18_diana.gmodule)  
serializer.dump(module_descriptions_pth) 
print("quant init")
#x ,_ = train_dataloader.dataset.__getitem__(0) 
#x = x.unsqueeze(0) 
#resnet18_diana.start_observing()
#_ = resnet18_diana(x) 
#resnet18_diana.stop_observing()

trainer = pl.Trainer(accelerator='gpu', devices=[0])
resnet18_diana.initialize_quantization(trainer)
#
model_save_path = Path("zoo/imagenet/resnet18/quantized/weights.pth")
#resnet18_diana.gmodule.load_state_dict(torch.load(model_save_path)["state_dict"]) 
torch.save ({
                       'state_dict': resnet18_diana.gmodule.state_dict(),
                    } , model_save_path)
print("quant init finished ")
resnet18_diana.freeze_clipping_bound() 

#training 
checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath='zoo/imagenet/resnet18/FQ/',filename='imgnet-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False  )
callbacks = [checkpoint_callback,EarlyStopping(monitor="val_acc", mode="max", patience=10) ]
max_epochs = 60
trainer = pl.Trainer(accelerator = "gpu",strategy="ddp",  devices = -1 , max_epochs =max_epochs , callbacks=callbacks,logger=CSVLogger(save_dir="logs/FQfrozen"))
print("starting FQ training") 
resnet18_diana.set_optimizer('SGD' , lr=0.00001 ,    weight_decay=1e-5)
#resnet18_diana.load_state_dict(torch.load("zoo/imagenet/resnet18/FQ/imgnet-epoch=07-val_acc=0.4658.ckpt")["state_dict"]) 
trainer.fit(resnet18_diana , train_dataloader , validation_dataloader)

resnet18_diana.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])

resnet18_diana.unfreeze_clipping_bound() 
print("finished FQ training with frozen bounds") 
resnet18_diana.set_optimizer('SGD' , lr=0.01)
print("starting FQ training with unfrozen bounds") 
max_epochs = 60 # train the bound for 10 epochs
checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath='zoo/imagenet/resnet18/FQ_bounds/',filename='imgnet-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
callbacks = [checkpoint_callback ,EarlyStopping(monitor="val_acc", mode="max", patience=10) ]
trainer = pl.Trainer(accelerator = "gpu", strategy = "ddp" , devices = -1 ,max_epochs =max_epochs , callbacks=callbacks,logger=CSVLogger(save_dir="logs/FQunfrozen"))

trainer.fit(resnet18_diana , train_dataloader , validation_dataloader)
resnet18_diana.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
print("finished FQ training with unfrozen bounds") 
#
#hw mapping 
print("mapping to hw")
resnet18_diana.map_to_hw() 
print("finished mapping to hw")
#
#
#Re-training 
lr = 0.01 
momentum = 0.4
resnet18_diana.set_optimizer('SGD' , lr=lr , momentum=momentum, weight_decay=weight_decay)
checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath='zoo/imagenet/resnet18/HW_mapped/',filename='imgnet-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
callbacks = [checkpoint_callback ,EarlyStopping(monitor="val_acc", mode="max", patience=10) ]
max_epochs = 60
trainer = pl.Trainer(accelerator = "gpu", strategy = "ddp" , devices = -1 ,max_epochs =max_epochs , callbacks=callbacks,logger=CSVLogger(save_dir="logs/HWmapped"))
trainer.accelerator = "gpu"
trainer.strategy = "ddp"
trainer.devices = -1 
print("starting hw mapepd training ")
trainer.fit(resnet18_diana , train_dataloader , validation_dataloader)
print("finished hw mapped training ")
resnet18_diana.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
#
#
#end 
print("integrizing layers")
resnet18_diana.integrize_layers()
print("finished integrizing layers")
model_save_path = Path("zoo/imagenet/resnet18/HW_mapped/weights.pth")
torch.save ({
                        'state_dict': resnet18_diana.state_dict(),
                    } , model_save_path)
#
data_folder = Path("backend/imgnet/resnet18")
resnet18_diana.export_model(str(data_folder.absolute()))