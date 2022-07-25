import torch 
from torch import nn 

class DigitalRequantizer(nn.Module): # div and clip operations 
    def __init__(self, scale : torch.Tensor , zero : torch.Tensor , n_levels : torch.Tensor) -> None:
        # scale and clipping range 
        super().__init__() 
        self.register_buffer("div", torch.Tensor()) # scale 
        self.register_buffer("zero", torch.Tensor())
        self.register_buffer("n_levels", torch.Tensor())

        self.qop = DigitalQuantOp.apply
        
    def forward(self , x : torch.Tensor): 
        return self.qop(self.div,self.zero, self.n_levels) 
     
# to ensure correct format in onnx file 
class DigitalQuantOp(torch.autograd.Function):
    @staticmethod 
    def forward(ctx , x : torch.Tensor, div : torch.Tensor, zero : torch.Tensor , n_levels: torch.Tensor): 
        quantized_x = x / div 
        return torch.clip(quantized_x.floor() , zero , n_levels-1) #clipped 
    
    @staticmethod
    def symbolic(g: torch._C.graph, x: torch._C.Value, div :torch._C.Value , zero : torch._C.Value , n_levels: torch.Tensor ) -> torch._C.Value:
        return g.op("Clip" ,g.op("Floor", g.op("Div", x , div )) , zero , n_levels -1) 
        #access graph variables using g.div
        pass
