import argparse
from functools import partial
import utils
import os
import copy
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as ds
import torchvision
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Loader import ModulesLoader


BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 2
BASE_LR = 0.0005


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.models.keys(), help="Model architecture")
parser.add_argument("weights", help="Model weights floating-point model (.pth), fake quantized model (.pth.fq) or hardware mapped model (.pth.hw)")
parser.add_argument("-c", "--config", help="Model config file (.yaml)", default=None)
parser.add_argument("-e", "--export-dir", help="Directory to export onnx and feature files", default='export')
args = parser.parse_args()

# enable determinism
torch.manual_seed(0)

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocess = utils.get_preprocess(args.model)

train_dataset = ds.CIFAR10(
    "./data/cifar10/train",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomHorizontalFlip(),
        ] + preprocess
    ),
)
val_dataset = ds.CIFAR10(
    "./data/cifar10/validation",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(preprocess),
)

# define the data loaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# define criterion
criterion = nn.CrossEntropyLoss()

# define criterion
optimizer_cls = partial(optim.Adam, lr=BASE_LR)
lr_scheduler_cls = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=EPOCHS)

# define model and load weights from fp training
model = utils.models[args.model]()

# define a calibration dataset for the quantization parameters
def representative_dataset():
    for _, (data, _) in zip(range(5), val_dataloader):
        yield data

weights_file = args.weights

if weights_file.endswith('.pth'):
    # load full-precision model weights
    sd = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(sd["net"])
    target_acc = sd['acc']  # try to reach the same accuracy as the floating point model
    print("Floating-point accuracy = %.3f"%(100 * target_acc))

module_description = None
if args.config is not None:
    loader = ModulesLoader()
    module_description = loader.load(args.config)

# wrap model in quantlib wrapper
fq_model = DianaModule(
    DianaModule.from_trainedfp_model(
        model, modules_descriptors=module_description,
    ),
    representative_dataset,
)

# quantize weights and activation
fq_model.set_quantized(activations=True)

if weights_file.endswith('.pth'):
    # train model with quantized weights and activations
    print("Start FQ training with quantized weights and activations")
    weights_file = weights_file.replace('.pth', '.pth.fq')
    utils.train_classifier(fq_model, train_dataloader, val_dataloader, optimizer_cls,
                           criterion, lr_scheduler_cls, EPOCHS, device,
                           weights_file, target_acc)
    fq_model = fq_model.to('cpu')

if weights_file.endswith('.pth.fq'):
    # load fake quantized model weights
    sd = torch.load(weights_file, map_location="cpu")
    fq_model.load_state_dict(sd["net"])
    target_acc = sd['acc']  # try to reach the same accuracy as the floating point model
    print("Fake quantized accuracy = %.3f"%(100 * target_acc))

# map to hardware (also folds batchnorm layers)
fq_model.map_to_hw()

if weights_file.endswith('.pth.fq'):
    # train model with quantized weights and activations
    print("Start HW mapped training")
    weights_file = weights_file.replace('.pth.fq', '.pth.hw')
    utils.train_classifier(fq_model, train_dataloader, val_dataloader, optimizer_cls,
                           criterion, lr_scheduler_cls, EPOCHS, device,
                           weights_file, target_acc)
    fq_model = fq_model.to('cpu')

if weights_file.endswith('.pth.hw'):
    # load fake quantized model weights
    sd = torch.load(weights_file, map_location="cpu")
    fq_model.load_state_dict(sd["net"])
    target_acc = sd['acc']  # try to reach the same accuracy as the floating point model
    print("HW mapped quantized accuracy = %.3f"%(100 * target_acc))

# convert all decimal values to integer values (still float dtype)
fq_model.integrize_layers()

# validation of PTQ model accuracy
fq_model = fq_model.to(device)
acc = utils.validation(fq_model, val_dataloader, criterion, device)
print("Integrized accuracy = %.3f"%(100 * acc))

# export to ONNX file
fq_model = fq_model.to('cpu')
fq_model.export_model(args.export_dir)
