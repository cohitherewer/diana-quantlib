import torch
import argparse
from functools import partial
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
import dataset


BATCH_SIZE = 100
EPOCHS = 36
NUM_WORKERS = 4
BASE_LR = 0.0005


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=utils.audio_classifier_models.keys(), help="Model architecture")
parser.add_argument("weights", help="File to store the best model weights (.pth)")
args = parser.parse_args()

# use cuda if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = dataset.KeywordSpottingDataset(
    "./data",
    download=True,
    subset="training"
)
print(len(train_dataset))

val_dataset = dataset.KeywordSpottingDataset(
    "./data",
    download=True,
    subset="validation"
)
print(len(val_dataset))

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
lr_scheduler_cls = partial(optim.lr_scheduler.MultiStepLR, milestones=[12, 24, 36])

# define model
model = utils.audio_classifier_models[args.model]()

print("Start FP training")
target_acc = 1.0    # no limit on accuracy
# define optimizer and learning rate scheduler (according to mlpef specs)
acc = utils.train_classifier(model, train_dataloader, val_dataloader, optimizer_cls,
                                    criterion, lr_scheduler_cls, EPOCHS, device,
                                    args.weights, target_acc)
