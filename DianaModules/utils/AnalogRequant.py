import torch 
from torch import nn 

class AnalogRequantizer(nn.Module): # div and clip operations 
    def __init__(self, scale : torch.Tensor , clip : torch) -> None:
        # scale and clipping range 
        super().__init__() 
        self.register_buffer("div", torch.Tensor()) # scale 
        self.register_buffer("zero", torch.Tensor())
        self.register_buffer("n_levels", torch.Tensor())
        self.register_buffer("mul", torch.Tensor())
        self.register_buffer("add", torch.Tensor())
        self.qop = AnalogQuantOp.apply
        
    def forward(self , x : torch.Tensor): 
        return self.qop(self.div,self.zero, self.n_levels ,self.mul , self.add ) 
     

class AnalogQuantOp(torch.autograd.Function):
    @staticmethod 
    def forward(ctx , x : torch.Tensor, div : torch.Tensor, zero : torch.Tensor , n_levels: torch.Tensor, mul : torch.Tensor, add : torch.Tensor): 
        pass
    
    @staticmethod
    def symbolic(g: torch._C.graph, x: torch._C.Value, div :torch._C.Value , zero : torch._C.Value , n_levels: torch.Tensor ,mul :torch._C.Value , add : torch._C.Value) -> torch._C.Value:
        
        #access graph variables using g.div
        pass
