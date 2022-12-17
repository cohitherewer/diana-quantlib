from enum import Enum, auto
from typing import Optional, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics  
# * Knowledge Distillation
# * Note: error from teach and true prediction are combined

class OptimizerType(Enum):
    SGD = auto()
    ADAM = auto()


class LrScheduler(Enum):
    COSINE = auto()
    MULTISTEP = auto()
    EXP = auto()
    NONE = auto()


class DistilLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temp: float = 1.0):
        super(DistilLoss, self).__init__()
        self.alpha = alpha
        self.T = temp

    def forward(self, distill_outputs, target):
        student_logits, teacher_logits = distill_outputs
        p = nnf.log_softmax(student_logits / self.T, dim=1)
        q = nnf.softmax(teacher_logits / self.T, dim=1)
        l_kl = nnf.kl_div(p, q, reduction='sum') * (self.T ** 2) / student_logits.shape[0]
        l_ce = nnf.cross_entropy(student_logits, target)
        return l_kl * self.alpha + l_ce * (1. - self.alpha)


class QModelDistiller(pl.LightningModule):

    def __init__(
            self, student: nn.Module, teacher: nn.Module,
            learning_rate: float, warm_up: int, momentum: float, max_epochs: int,
            weight_decay: float, nesterov: bool, lr_scheduler: str, gamma: float, optimizer: str,
            seed: int, version: str = None
    ):

        super().__init__()
        self.save_hyperparameters(
            'learning_rate', 'warm_up', 'momentum', 'weight_decay', 'nesterov', 'lr_scheduler', 'gamma',
            'optimizer', 'max_epochs', 'seed', 'version'
        )
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        self.criterion = DistilLoss(0.5, 1.0)

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc  = torchmetrics.Accuracy()

    def forward(self, x):
        return self.student(x)

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:

        if self.trainer.global_step < self.hparams.warm_up:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warm_up)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        super(QModelDistiller, self).optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu,
            using_native_amp,
            using_lbfgs)

    def training_step(self, batch, batch_idx):
        self.teacher.eval()  # validation/train switch can set teacher to train
        x, y = batch
        y_comp = self.student(x)
        with torch.no_grad():
            teacher_y = self.teacher(x)
        y_hat = (y_comp, teacher_y)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss , prog_bar=True)
        # acc 
        return {"loss": loss, "pred":y_comp, "true":y}

    def training_step_end(self, outputs) :
        self.train_acc(outputs["pred"] ,outputs["true"] )
        self.log("train_acc" , self.train_acc ,on_step=False,  on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.student(x)
        loss = nnf.cross_entropy(y_hat, y)

        self.log("val_loss", loss , prog_bar=True ,sync_dist=True)
        return {"loss": loss, "pred":y_hat, "true":y}
    def validation_step_end(self, outputs) :
        self.valid_acc(outputs["pred"] ,outputs["true"] )
        self.log("val_acc" ,self.valid_acc, on_step=False,  on_epoch=True , prog_bar=True, sync_dist=True)
        return outputs["loss"]

    def load_weights(self, weights_file: str):
        self.student.load_state_dict(torch.load(weights_file), strict=False)

    def save_weights(self, weights_file: str):
        torch.save(self.student.state_dict(), weights_file)

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = nnf.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.test_acc(nnf.softmax(y_hat, dim=-1), y)
        self.log('test_acc', self.test_acc)
        return loss

    def configure_optimizers(self):
        try:
            opt_type = OptimizerType[self.hparams.optimizer.upper()]
        except KeyError:
            raise RuntimeError(f"Unknown optimizer type {self.hparams.optimizer}")

        try:
            sched_type = LrScheduler[self.hparams.lr_scheduler.upper()]
        except KeyError:
            raise RuntimeError(f"Unknown optimizer type {self.hparams.lr_scheduler}")

        if opt_type is OptimizerType.SGD:
            opts = [torch.optim.SGD(
                self.student.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                nesterov=self.hparams.nesterov
            )]
        elif opt_type is OptimizerType.ADAM:
            opts = [torch.optim.Adam(
                self.student.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )]
        else:
            raise RuntimeError(f"Unsupported Optimizer type: {self.hparams.optimizer}")
        
        if sched_type is LrScheduler.COSINE:
            schedulers = [CosineAnnealingLR(opts[-1], T_max=self.hparams.max_epochs, eta_min=0.)]
        elif sched_type is LrScheduler.MULTISTEP:
            # decay lr at 30%, 60% and 80%
            milestones = [int(r * self.hparams.max_epochs) for r in (0.3, 0.6, 0.8)]
            schedulers = [MultiStepLR(opts[-1], milestones=milestones, gamma=self.hparams.gamma)]
        elif sched_type is LrScheduler.EXP: 
            # Patience of 5 and a factor of 0.1
            schedulers = [ReduceLROnPlateau(opts[-1] , mode='max',patience=5)] 
            return {"optimizer": opts[0] ,"lr_scheduler": schedulers[0], "monitor": 'val_acc'} 
            pass 
        else:
            schedulers = []

        return opts, schedulers