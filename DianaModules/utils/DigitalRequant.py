
import torch 
from torch import nn

from quantlib.editing.graphs.nn.requant import Requantisation 

class DigitalRequantizer(Requantisation): # div and clip operations # neeed to inhereit from requantisation for tracing and traversal of quantlib 
    def __init__(self, scale : torch.Tensor , zero : torch.Tensor , n_levels : torch.Tensor) -> None:
        # scale and clipping range 
        nn.Module.__init__(self)
        self.register_buffer("div", scale) # scale 
        self.register_buffer("zero", zero)
        self.register_buffer("clip_hi", n_levels-1)

        self.qop = DigitalQuantOp.apply
        
    def forward(self , x : torch.Tensor) -> torch.Tensor: 
        return self.qop(x, self.div,self.zero, self.clip_hi) 
     
# to ensure correct format in onnx file 
class DigitalQuantOp(torch.autograd.Function):
    @staticmethod 
    def forward(ctx , x : torch.Tensor, div : torch.Tensor, zero : torch.Tensor , clip_hi: torch.Tensor): 
        ctx.save_for_backward(x)    
        quantized_x = x / div 
        return torch.clip(torch.floor(quantized_x ), min=zero , max =clip_hi) #clipped 
    
    @staticmethod 
    def symbolic(g: torch._C.Graph, x: torch._C.Value, div: torch._C.Value, zero: torch._C.Value, clip_hi: torch._C.Value) -> torch._C.Value:
        return g.op("Min", g.op("Max" ,g.op("Floor", g.op("Div", x , div )) ,zero), clip_hi) 
      
      