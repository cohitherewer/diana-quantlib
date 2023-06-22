import torch
from torch import nn, optim

import pytorch_lightning as pl
import torchmetrics
from dianaquantlib.utils.BaseModules import DianaModule
import importlib


class DianaLightningModule(pl.LightningModule):
    def __init__(
        self,
        diana_module: DianaModule = None,
        criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.diana_module = diana_module
        self.optimizer = None
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.criterion = criterion

    def forward(self, x: torch.Tensor):
        return self.diana_module(x)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        if self.diana_module._integrized:
            x = torch.floor(x / self.dataset_scale.to(x.device))
        yhat = self.diana_module(x)
        loss = self.criterion(yhat, y)
        self.log("train_loss", loss, prog_bar=True)
        # acc
        return {"loss": loss, "pred": yhat, "true": y}

    def training_step_end(self, outputs):
        self.train_acc(outputs["pred"], outputs["true"])
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return outputs["loss"]

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        if self.diana_module._integrized:
            x = torch.floor(x / self.dataset_scale.to(x.device))
        yhat = self.diana_module(x)

        loss = self.criterion(yhat, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "pred": yhat, "true": y}

    def validation_step_end(self, outputs):
        self.valid_acc(outputs["pred"], outputs["true"])
        self.log(
            "val_acc",
            self.valid_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return outputs["loss"]

    def set_optimizer(self, type="SGD", *args, **kwargs):  # case sensitive
        my_module = importlib.import_module("torch.optim")
        MyClass = getattr(my_module, type)
        self.optimizer = MyClass(
            self.diana_module.parameters(), *args, **kwargs
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5
        )

    def configure_optimizers(self):
        self.set_optimizer("SGD", lr=0.01)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_acc",
            },
        }
