import os
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as ds
import torchvision
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Loader import ModulesLoader

import utils
import dataset


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.anomaly_models.keys(), help="Model architecture")
parser.add_argument("weights", help="Model weights floating-point model (.pth)")
parser.add_argument("-c", "--config", help="Model config file (.yaml)", default=None)
parser.add_argument("-e", "--export-dir", help="Directory to export onnx and feature files", default='export')
args = parser.parse_args()

if not os.path.exists(args.export_dir):
    os.makedirs(args.export_dir)

# enable determinism
torch.manual_seed(0)

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define validation set for testing and select calibration data
val_dataset = dataset.AnomalyMNIST(
    "./data/mnist/validation",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose(utils.get_preprocess(args.model)),
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
criterion = nn.MSELoss()

# define model
model = utils.anomaly_models[args.model]()

# define a calibration dataset for the quantization parameters
def representative_dataset():
    for _, (data, _) in zip(range(5), val_dataloader):
        yield data

# load model
sd = torch.load(args.weights, map_location="cpu")
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

# map to hardware (also folds batchnorm layers)
fq_model.map_to_hw()

# convert all decimal values to integer values (still float dtype)
fq_model.integrize_layers()

# uncomment to show the final quantized graph in pytorch before export
#print("After Integrize ----------")
#print(fq_model.gmodule)

# validation of PTQ model accuracy
fq_model = fq_model.to(device)
acc = utils.validation_anomaly(fq_model, val_dataloader, criterion, device)
print("Quantized accuracy = %.3f"%(100 * acc))

# export to ONNX file
fq_model = fq_model.to('cpu')
fq_model.export_model(args.export_dir)
