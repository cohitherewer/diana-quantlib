from turtle import forward

import torch
from torch import nn

from quantlib.editing.graphs.nn.requant import Requantisation


class MulAdd(Requantisation):
    def __init__(self, mul: torch.Tensor, add: torch.Tensor) -> None:
        # scale and clipping range
        nn.Module.__init__(self)
        self.register_buffer("mul", mul)  # scale
        self.register_buffer("add", add)

    # self.qop = MulAddOp.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.set_detect_anomaly(True):
            return self.mul * x + self.add
        # return self.qop(x,self.mul , self.add )


# to ensure correct format in onnx file
class MulAddOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mul: torch.Tensor, add: torch.Tensor):
        # mul.reshape(x.shape)
        # add.reshape(x.shape)
        return x * mul + add

    @staticmethod
    def symbolic(
        g: torch._C.Graph,
        x: torch._C.Value,
        mul: torch._C.Value,
        add: torch._C.Value,
    ) -> torch._C.Value:

        # return g.op("Min", g.op("Max" ,g.op("Floor", g.op("Div", x , g.op("Constant", value_t=torch.tensor(div, dtype=torch.float) ))) ,zero), clip_hi)
        return g.op("Add", add, g.op("Mul", x, mul))
