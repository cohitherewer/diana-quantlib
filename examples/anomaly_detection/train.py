import utils
import numpy as np
import random
import torch
import argparse
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as ds
from torch.utils.data import DataLoader
import torchvision


BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 4
BASE_LR = 0.001


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.anomaly_models.keys(), help="Model architecture")
parser.add_argument("weights", help="File to store the best model weights (.pth)")
args = parser.parse_args()

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocess = utils.get_preprocess(args.model)

train_dataset = ds.MNIST(
    "./data/mnist/train",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
        ] + preprocess
    ),
)
val_dataset = utils.AnomalyMNIST(
    "./data/mnist/validation",
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
criterion = nn.MSELoss()

# define criterion
optimizer_cls = partial(optim.Adam, lr=BASE_LR)
lr_scheduler_cls = partial(optim.lr_scheduler.CosineAnnealingLR, T_max=EPOCHS)

# define model
model = utils.anomaly_models[args.model]()

print("Start FP training")
target_acc = 1.0    # no limit on accuracy
# define optimizer and learning rate scheduler (according to mlpef specs)
acc = utils.train_anomaly(model, train_dataloader, val_dataloader, optimizer_cls,
                          criterion, lr_scheduler_cls, EPOCHS, device,
                          args.weights, target_acc)
