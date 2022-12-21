import torchvision.datasets as ds
import torchvision
import pytorch_lightning as pl
import torch
from pathlib import Path
from DianaModules.utils.BaseModules import DianaModule

from DianaModules.utils.serialization.Loader import ModulesLoader
from torch.utils.data import DataLoader
from DianaModules.utils.serialization.Serializer import ModulesSerializer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from DianaModules.models.cifar10.FastResnet import resnet8
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from test import load_extra

train_dataset = ds.CIFAR10(
    "./data/cifar10/train",
    train=True,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    ),
)
test_dataset = ds.CIFAR10(
    "./data/cifar10/validation",
    train=False,
    download=False,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    ),
)

train_dataloader = DataLoader(
    train_dataset,
    pin_memory=True,
    num_workers=28,
    shuffle=False,
    batch_size=256,
)
val_dataloader = DataLoader(
    test_dataset, pin_memory=True, num_workers=28, batch_size=256
)
train_scale = torch.Tensor([0.03125])  # pow2
output_weights_path = "zoo/cifar10/resnet8"
FP_weights = output_weights_path + "/fp.pth"
# model = Network(net()) # Floating point
fp = resnet8()
fp.load_state_dict(torch.load(FP_weights, map_location="cpu")["state_dict"])
fp.eval()

# ---------------------------------------------------------------------------- #
#                            Floating point training                           #
# ---------------------------------------------------------------------------- #

# print("training floating point model ")
# fp = DianaModule(model)
# fp.load_state_dict(torch.load("zoo/cifar10/resnet8/resnet8-epoch=71-val_acc=0.8149.ckpt", map_location="cpu")["state_dict"])
# fp.set_optimizer('SGD', lr =0.01 , momentum=0.4 )
# checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath=output_weights_path,filename='resnet8-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
# callbacks = [checkpoint_callback  ]
# max_epochs = 120
# trainer = pl.Trainer(accelerator = "gpu",devices = [0])
# trainer.validate(fp, val_dataloader)

# ---------------------------------------------------------------------------- #
#                               Fake Quantization                              #
# ---------------------------------------------------------------------------- #
module_descriptions_pth = (
    "/imec/other/csainfra/nada64/DianaTraining/serialized_models/resnet8.yaml"
)

# region ModuleLoader
loader = ModulesLoader()
module_descriptions = loader.load(module_descriptions_pth)

model = DianaModule(DianaModule.from_trainedfp_model(fp, module_descriptions))

model.attach_train_dataloader(train_dataloader, scale=train_scale)
model.attach_quantization_dataloader(train_dataloader)
model.attach_validation_dataloader(val_dataloader, scale=train_scale)
serializer = ModulesSerializer(model.gmodule)
serializer.dump(module_descriptions_pth)
# Initializing quantization
# trainer = pl.Trainer(accelerator='gpu' , devices=[0])

# model.initialize_quantization(trainer)  #When intiializing the quantization make sure you pass a trainer that has only 1 device. All others can be run on multiple devices. (check the imagenet training for an example)
model_save_path = Path("zoo/cifar10/resnet8/quantized_mixed.pth")
model.set_quantized()
model.load_state_dict(
    torch.load(
        "zoo/cifar10/resnet8/resnet8_fq_mixed-epoch=13-val_acc=0.9247.ckpt",
        map_location="cpu",
    )["state_dict"]
)
# torch.save ({
#                       'state_dict': model.gmodule.state_dict(),
#                    } , model_save_path)
# print("finished quantization")
# trainer.teardown()
# model.freeze_clipping_bound()
# model.set_optimizer('SGD', lr = 0.01)
##
# checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath=output_weights_path,filename='resnet8_fq_mixed-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
# callbacks = [checkpoint_callback  ]
# max_epochs = 5
# trainer = pl.Trainer(accelerator = "gpu" ,  devices = [0] , max_epochs =max_epochs , callbacks=callbacks)
# trainer.fit(model, train_dataloaders=train_dataloader ,val_dataloaders=val_dataloader )
# model.unfreeze_clipping_bound()
# checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath=output_weights_path,filename='resnet8_fq_act_mixed-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
# max_epochs = 24
# trainer = pl.Trainer(accelerator = "gpu",  devices = [0] , max_epochs =max_epochs , callbacks=callbacks)
# trainer.fit(model, train_dataloaders=train_dataloader ,val_dataloaders=val_dataloader )

print("finished training model")
print("starting hw mapping")
model.map_to_hw()

model.load_state_dict(
    torch.load(
        "zoo/cifar10/resnet8/resnet8_hw_mixed-epoch=73-val_acc=0.9017.ckpt",
        map_location="cpu",
    )["state_dict"]
)
print("finshed hw mapping")
# model.set_optimizer("SGD", lr =0.01)
# checkpoint_callback = ModelCheckpoint(monitor = "val_acc" ,mode="max",  dirpath=output_weights_path,filename='resnet8_hw_mixed-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False)
# max_epochs = 120
# callbacks = [checkpoint_callback  ]
# trainer = pl.Trainer(accelerator = "gpu",  devices = [0] , max_epochs =max_epochs , callbacks=callbacks)
# trainer.fit(model, train_dataloaders=train_dataloader ,val_dataloaders=val_dataloader )
model.to("cpu")
model.integrize_layers()
print("finshed layer integrization")

#####
#####
trainer = pl.Trainer(accelerator="gpu", devices=[1])
trainer.validate(model, val_dataloader)
# data_folder = Path("backend/cifar10/resnet8")
# model.export_model(str(data_folder.absolute()))
#####
