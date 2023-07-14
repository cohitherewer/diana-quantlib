import os
import argparse
from functools import partial
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as ds
import torchvision
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Loader import ModulesLoader

import utils
import dataset

BATCH_SIZE = 100
EPOCHS = 20
NUM_WORKERS = 4
BASE_LR = 0.0003


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.audio_classifier_models.keys(), help="Model architecture")
parser.add_argument("weights", help="Model weights floating-point model (.pth), fake quantized model (.pth.fq) or hardware mapped model (.pth.hw)")
parser.add_argument("-c", "--config", help="Model config file (.yaml)", default=None)
parser.add_argument("-e", "--export-dir", help="Directory to export onnx and feature files", default='export')
args = parser.parse_args()

if not os.path.exists(args.export_dir):
    os.makedirs(args.export_dir)

# enable determinism
torch.manual_seed(0)

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = dataset.KeywordSpottingDataset(
    "./data",
    download=True,
    subset="training"
)

val_dataset = dataset.KeywordSpottingDataset(
    "./data",
    download=True,
    subset="validation"
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
    shuffle=True,
)

# define criterion
criterion = nn.CrossEntropyLoss()

# define criterion
optimizer_cls = partial(optim.Adam, lr=BASE_LR)
lr_scheduler_cls = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=EPOCHS)

# define model and load weights from fp training
model = utils.audio_classifier_models[args.model]()

# define a calibration dataset for the quantization parameters
def representative_dataset():
    for _, (data, _) in zip(range(2), val_dataloader):
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
        qhparamsinitstrategy='meanstd'
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
acc = utils.validation_classifier(fq_model, val_dataloader, criterion, device)
print("Integrized accuracy = %.3f"%(100 * acc))

# export to ONNX file
fq_model = fq_model.to('cpu')
fq_model.export_model(args.export_dir)
