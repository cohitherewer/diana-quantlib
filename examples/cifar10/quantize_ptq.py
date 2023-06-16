import os
import utils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as ds
import torchvision
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.serialization.Loader import ModulesLoader
from DianaModules.models.mlperf_tiny import DSCNN, DAE, MobileNet, ResNet
from DianaModules.models.cifar10.cifarnet import CifarNet8
from DianaModules.utils.serialization.Serializer import ModulesSerializer


DEVICE = 'cuda'
MODEL_FP_WEIGHTS_PATH = './checkpoints/best_fp_model.pth'
MODEL_EXPORT_DIR = './export'
MODEL_CONFIG_PATH = './config/cifarnet8.yaml'

# enable determinism
torch.manual_seed(0)

# define validation set for testing and select calibration data
val_dataset = ds.CIFAR10(
    "./data/cifar10/validation",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),              # convert to range  [0, 1]
            torchvision.transforms.Normalize((0.5), (0.5)), # convert to range [-1, 1]
        ]
    ),
)

# define the data loaders
val_dataloader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
)

# define criterion
criterion = nn.CrossEntropyLoss()

# define model
model = CifarNet8()

# define a calibration dataset for the quantization parameters
def representative_dataset():
    for _, (data, _) in zip(range(5), val_dataloader):
        yield data

# load model
sd = torch.load(MODEL_FP_WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(sd["net"])
target_acc = sd['acc']  # try to reach the same accuracy as the floating point model
print("Floating-point accuracy = %.3f"%(100 * target_acc))

#
module_description = None
if os.path.exists(MODEL_CONFIG_PATH):
    loader = ModulesLoader()
    module_description = loader.load(MODEL_CONFIG_PATH)

# wrap model in quantlib wrapper
fq_model = DianaModule(
    DianaModule.from_trainedfp_model(
        model, modules_descriptors=module_description,
        qhparamsinitstrategy='minmax'
    ),
    representative_dataset,
)

# quantize weights and activation
fq_model.set_quantized(activations=True)

# map to hardware (also folds batchnorm layers)
fq_model.map_to_hw()

# convert all decimal values to integer values (still float dtype)
fq_model.integrize_layers()

# uncomment to show the final quantized graph in pytorch before export
#print(fq_model.gmodule)

# validation of PTQ model accuracy
fq_model = fq_model.to(DEVICE)
acc = utils.validation(fq_model, val_dataloader, criterion, DEVICE)
print("Quantized accuracy = %.3f"%(100 * acc))

# export to ONNX file
fq_model = fq_model.to('cpu')
fq_model.export_model(MODEL_EXPORT_DIR)
