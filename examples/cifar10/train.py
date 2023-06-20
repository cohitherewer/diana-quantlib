import utils
import torch
import argparse
from functools import partial
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as ds
from torch.utils.data import DataLoader
import torchvision


BATCH_SIZE = 32
EPOCHS = 100
NUM_WORKERS = 2
BASE_LR = 0.001


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.models.keys(), help="Model architecture")
parser.add_argument("weights", help="File to store the best model weights (.pth)")
args = parser.parse_args()

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = ds.CIFAR10(
    "./data/cifar10/train",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),              # convert to range  [0, 1]
            torchvision.transforms.Normalize((0.5), (0.5)), # convert to range [-1, 1]
        ]
    ),
)
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

print("Start FP training")
target_acc = 1.0    # no limit on accuracy
# define optimizer and learning rate scheduler (according to mlpef specs)
acc = utils.train_classifier(model, train_dataloader, val_dataloader, optimizer_cls,
                                    criterion, lr_scheduler_cls, EPOCHS, device,
                                    args.weights, target_acc)
