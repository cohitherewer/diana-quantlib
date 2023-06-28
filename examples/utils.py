import torch
import torchvision
from functools import partial
from tqdm import tqdm
from sklearn import metrics
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.models.mlperf_tiny import ResNet, MobileNet, DAE, DSCNN
from dianaquantlib.models.cifar10.cifarnet import CifarNet8

image_classifier_models = {
    'resnet': ResNet,
    'cifarnet8': CifarNet8,
    'mobilenet': partial(MobileNet, num_classes=12),
}

audio_classifier_models = {
    'dscnn': DSCNN,
}

anomaly_models = {
    'dae': partial(DAE, num_outputs=28*28),
}

all_models = image_classifier_models | audio_classifier_models | anomaly_models


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

    if model_name == 'dae':
        preprocess += [
            torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
        ]

    return preprocess


class ClassifierAccuracy():

    def __init__(self):
        self.total = 0
        self.correct = 0

    def __call__(self, outputs, targets):
        _, predicted = outputs.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
        acc = self.correct / self.total
        return acc


def train_epoch_classifier(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    calc_accuracy = ClassifierAccuracy()
    progress_bar = tqdm(dataloader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        acc = calc_accuracy(outputs, targets)
        progress_bar.set_description('Train Loss: %.3f | Train Acc: %.3f%%'% (train_loss / (batch_idx + 1), 100. * acc))


def validation_classifier(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0
    calc_accuracy = ClassifierAccuracy()
    progress_bar = tqdm(dataloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
            acc = calc_accuracy(outputs, targets)
            progress_bar.set_description('Val Loss: %.3f | Val Acc: %.3f%%'% (validation_loss / (batch_idx + 1), 100. * acc))

    return acc


def train_epoch_anomaly(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    progress_bar = tqdm(dataloader)
    for batch_idx, (inputs, _) in enumerate(progress_bar):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_description('Train Loss: %.3f'%(train_loss / (batch_idx + 1)))


def validation_anomaly(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0
    progress_bar = tqdm(dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            prediction = torch.mean(torch.square(outputs - inputs), dim=1)  # calculate MSE error
            predictions.append(prediction.cpu())
            labels.append(targets)
            validation_loss += loss.item()
            progress_bar.set_description('Val Loss: %.3f'% (validation_loss / (batch_idx + 1)))

    # Calculate ROC AUC score
    labels = torch.cat(labels).numpy()
    predictions = torch.cat(predictions).numpy()
    auc = metrics.roc_auc_score(labels, predictions)
    print("Accuracy (ROC UAC) = %.3f%%"% (100 * auc))
    return auc


def train_model(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc,
                train_fn, val_fn):

    print("Start training, early stopping accuracy = %.3f"%(100 * early_stop_acc))
    best_acc = 0.0
    model = model.to(device)    # ensure the model is on the device

    # Validation before training
    acc = val_fn(model, val_dataloader, criterion, device)
    print("Starting accuracy = %.3f"%(100 * acc))

    optimizer = optimizer_cls(model.parameters())
    lr_scheduler = lr_scheduler_cls(optimizer)
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: %d, lr: %f' % (epoch + 1, lr))

        train_fn(model, train_dataloader, optimizer, criterion, device)
        acc = val_fn(model, val_dataloader, criterion, device)
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


def train_classifier(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                     lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc=1.0):

    return train_model(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                       lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc,
                       train_epoch_classifier, validation_classifier)


def train_anomaly(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                  lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc=1.0):

    return train_model(model, train_dataloader, val_dataloader, optimizer_cls, criterion,
                       lr_scheduler_cls, epochs, device, best_model_path, early_stop_acc,
                       train_epoch_anomaly, validation_anomaly)
