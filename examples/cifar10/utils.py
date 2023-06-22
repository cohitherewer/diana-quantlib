import torch
import torchvision
from functools import partial
from tqdm import tqdm
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.models.mlperf_tiny import ResNet, MobileNet
from dianaquantlib.models.cifar10.cifarnet import CifarNet8

models = {
    'resnet': ResNet,
    'cifarnet8': CifarNet8,
    'mobilenet': partial(MobileNet, num_classes=12),
}

def get_preprocess(model_name):
    preprocess = []

    if model_name == 'mobilenet':
        preprocess += [
            torchvision.transforms.Resize(96)
        ]

    preprocess += [
        torchvision.transforms.ToTensor(),              # convert to range  [0, 1]
        torchvision.transforms.Normalize((0.5), (0.5)), # convert to range [-1, 1]
    ]

    return preprocess


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = correct / total
        progress_bar.set_description('Train Loss: %.3f | Train Acc: %.3f%%'% (train_loss/(batch_idx+1), 100. * acc))


def validation(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total
            progress_bar.set_description('Val Loss: %.3f | Val Acc: %.3f%%'% (validation_loss/(batch_idx+1), 100. * acc))

    return acc


def train_classifier(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                     lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc=1.0):

    print("Start training, early stopping accuracy = %.3f"%(100 * early_stop_acc))
    best_acc = 0.0
    model = model.to(device)    # ensure the model is on the device

    # Validation before training
    acc = validation(model, val_dataloader, criterion, device)
    print("Starting accuracy = %.3f"%(100 * acc))

    optimizer = optimizer_cls(model.parameters())
    lr_scheduler = lr_scheduler_cls(optimizer)
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: %d, lr: %f' % (epoch + 1, lr))

        train(model, train_dataloader, optimizer, criterion, device)
        acc = validation(model, val_dataloader, criterion, device)
        lr_scheduler.step()

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, best_model_path)
            best_acc = acc

        if acc > early_stop_acc:
            print("Accuracy %.3f reached early stopping accuracy %.3f"%(100 * acc, 100 * early_stop_acc))
            break

    return acc

