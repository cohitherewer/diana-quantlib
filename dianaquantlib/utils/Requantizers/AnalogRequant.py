import torch
from torch import nn

from quantlib.editing.graphs.nn.requant import Requantisation


class AnalogRequantizer(
    Requantisation
):  # # neeed to inhereit from requantisation for tracing and traversal of quantlib
    def __init__(
        self,
        div: torch.Tensor,
        zero: torch.Tensor,
        n_levels: torch.Tensor,
        mul: torch.Tensor,
        add: torch.Tensor,
    ):
        # scale and clipping range
        nn.Module.__init__(self)
        self.register_buffer("div", div)  # scale
        self.register_buffer("clip_lo", zero)
        self.register_buffer("clip_hi", zero + n_levels - 1)
        self.register_buffer("mul", mul)
        self.register_buffer("add", add)
        # self.qop = AnalogQuantOp.apply

    def forward(self, x: torch.Tensor):
        out = torch.floor((x * self.mul + self.add) / self.div)
        return torch.clip(out, min=self.clip_lo, max=self.clip_hi)

        # return self.qop(x , self.div,self.clip_lo, self.clip_hi,self.mul , self.add )


class AnalogQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        div: torch.Tensor,
        zero: torch.Tensor,
        clip_hi: torch.Tensor,
        mul: torch.Tensor,
        add: torch.Tensor,
    ):
        x = x * mul
        x = x + add
        quantized_x = x / div
        return torch.clip(torch.floor(quantized_x), min=zero, max=clip_hi)

    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        x: torch._C.Value,
        div: torch._C.Value,
        zero: torch._C.Value,
        clip_hi: torch.Tensor,
        mul: torch._C.Value,
        add: torch._C.Value,
    ) -> torch._C.Value:  # ensuring consistency in operators
        # mul add div floor clip (min + max)
        # return g.op("Min", g.op("Max" , g.op("Floor", g.op("Div" ,g.op("Add", g.op("Mul", x , mul)) ,add), div) , zero ) , clip_hi  )
        return g.op(
            "Min",
            g.op(
                "Max",
                g.op(
                    "Floor",
                    g.op("Div", g.op("Add", g.op("Mul", x, mul), add), div),
                ),
                zero,
            ),
            clip_hi,
        )
