import torch
from torch import nn

from quantlib.editing.graphs.nn.requant import Requantisation


class DigitalRequantizer(
    Requantisation
):  # div and clip operations # neeed to inhereit from requantisation for tracing and traversal of quantlib
    def __init__(
        self, div: torch.Tensor, zero: torch.Tensor, n_levels: torch.Tensor
    ) -> None:
        # scale and clipping range
        nn.Module.__init__(self)
        self.register_buffer("div", div)  # scale
        self.register_buffer("clip_lo", zero)
        self.register_buffer("clip_hi", n_levels - 1 + zero)
        self.floor = FloorOp.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.floor(x / self.div)
        return torch.clip(out, min=self.clip_lo, max=self.clip_hi)


class FloorOp(torch.autograd.Function):
    """ Autograd function that implements floor with a straight through estimator
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        """
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_y):
        """ Needed for QAT fine-tuning
        """
        return grad_y

    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        x: torch._C.Value,
    ) -> torch._C.Value:
        """ Needed for ONNX export
        """
        return g.op("Floor", x)
