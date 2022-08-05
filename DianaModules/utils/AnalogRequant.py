<<<<<<< HEAD
import torch 
from torch import nn

from quantlib.editing.graphs.nn.requant import Requantisation 

class AnalogRequantizer(Requantisation): # # neeed to inhereit from requantisation for tracing and traversal of quantlib 
    def __init__(self,scale : torch.Tensor , zero : torch.Tensor , n_levels : torch.Tensor, mul : torch.Tensor, add : torch.Tesnor ) -> None:
        # scale and clipping range 
        super().__init__() 
        self.register_buffer("div", scale ) # scale 
        self.register_buffer("clip_lo", zero)
        self.register_buffer("clip_hi", zero + n_levels-1)
        self.register_buffer("mul", mul)
        self.register_buffer("add", add )
        self.qop = AnalogQuantOp.apply
        
    def forward(self , x : torch.Tensor): 
        return self.qop(self.div,self.zero, self.clip_hi,self.mul , self.add ) 
     

class AnalogQuantOp(torch.autograd.Function):
    @staticmethod 
    def forward(ctx , x : torch.Tensor, div : torch.Tensor, zero : torch.Tensor , clip_hi : torch.Tensor, mul : torch.Tensor, add : torch.Tensor): 
        x = x * mul   
        x = x + add 
        quantized_x = x / div 
        return torch.clip(torch.floor(quantized_x ), min=zero , max =clip_hi) 

    @staticmethod
    def symbolic(g: torch._C.Graph, x: torch._C.Value, div :torch._C.Value , zero : torch._C.Value , clip_hi: torch.Tensor ,mul :torch._C.Value , add : torch._C.Value) -> torch._C.Value: # ensuring consistency in operators 
        # mul add div floor clip (min + max) 
        return g.op("Mul", g.op("Add" , g.op("Min", g.op("Max" ,g.op("Floor", g.op("Div", x , g.op("Constant", value_t=torch.tensor(div, dtype=torch.float) ))) ,zero), clip_hi) , add ) , mul  )
        
=======
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
    def symbolic(g: torch._C.Graph, x: torch._C.Value, div :torch._C.Value , zero : torch._C.Value , n_levels: torch.Tensor ,mul :torch._C.Value , add : torch._C.Value) -> torch._C.Value:
        
        #access graph variables using g.div
        pass
>>>>>>> 8cb6716aa4c536330e5d580ea4ac3c1a4cd190dd
