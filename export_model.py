import argparse
import os
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from DianaModules.core.Operations import AnalogConv2d
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.LightningModules import DianaLightningModule
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torchvision.datasets as ds
import torchvision
from DianaModules.utils.compression.ModelDistiller import QModelDistiller
from DianaModules.utils.compression.QuantStepper import QuantDownStepper
from DianaModules.utils.serialization.Loader import ModulesLoader
from DianaModules.utils.serialization.Serializer import ModulesSerializer
from DianaModules.core.Operations import AnalogOutIdentity
from DianaModules.models.mlperf_tiny import DSCNN, DAE, MobileNet, ResNet

# define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--export_dir",
    type=str,
    default="export/",
    help="the directory to export ONNX",
)
parser.add_argument(
    "--config_pth",
    type=str,
    default=None,
    help="the path to the model's quantization configuration file (yaml file) ",
)
# parse the arguments
args = parser.parse_args()
# define your Pytorch Lightning module

torch.manual_seed(1)

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        #self.c = nn.Conv2d(1, 1, 3, 1, 1)
        #self.bn = nn.BatchNorm2d(1)
        #self.r = nn.ReLU()
        self.net = nn.Sequential(
            #nn.Conv2d(1, 8, 3, 1, 1),
            #nn.BatchNorm2d(8),
            #nn.ReLU(),
            #nn.AdaptiveAvgPool2d((1, 1)),
            #nn.Flatten(),
            nn.Linear(8, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        )
        #self.c2 = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        #x = self.r(self.bn(self.c(x)))
        #x = x + self.net(x)
        #return self.c2(x)
        return self.net(x)

    def get_random_input(self):
        #return torch.randn([1, 1, 32, 32])
        return torch.randn([1, 8])


# instantiate the module
#module = MyModel()
module = MobileNet()
#module = DSCNN()
#module = ResNet()
#module = DAE()

dataset_item = module.get_random_input()
dataset_scale = torch.tensor(.5)

# load configurations file
module_descriptions_pth = args.config_pth
module_description = None
if module_descriptions_pth:
    loader = ModulesLoader()
    module_description = loader.load(module_descriptions_pth)

# fake-quantize model and attach scales
fq_module = DianaModule(
    DianaModule.from_trainedfp_model(
        module, modules_descriptors=module_description
    ),
)

serializer = ModulesSerializer(fq_module.gmodule)
serializer.dump("my_model.yaml")

print("After from_trainedfp_model -----------------------------------")
for _, module in fq_module.named_modules():
    print(module)
print("end -----------------------------------")

# load fq model
fq_module._input_shape = dataset_item.shape[1:]
fq_module.set_quantized(
    activations=True,
    dataset_item=dataset_item,
)

print("After set quantized -----------------------------------")
for _, module in fq_module.named_modules():
    print(module)
print("end -----------------------------------")


# map to hw
fq_module.map_to_hw(
    dataset_scale=dataset_scale,
    dataset_item=dataset_item,
)

print("After map HW -----------------------------------")
for _, module in fq_module.named_modules():
    print(module)
print("end -----------------------------------")

# export
os.makedirs(args.export_dir, exist_ok=True)
fq_module.integrize_layers(
    dataset_scale=dataset_scale,
    dataset_item=dataset_item,
)

print("After integrize -----------------------------------")
for _, module in fq_module.named_modules():
    print(module)
print("end -----------------------------------")

fq_module.export_model(
    args.export_dir,
    dataset_scale=dataset_scale,
    dataset_item=dataset_item,
)
