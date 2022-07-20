import torch

class _FakeDQuantiser(torch.autograd.Function): # symmetric quantisation 
    @staticmethod
    def forward(ctx,  x:   torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
   
        
        
        x = x / (step * scale) # quantized 
        x = torch.clip(x, clip_lo, clip_hi ) # clipping to bw
        x = torch.floor(x)
        
        x = x * (step * scale)
        return x 
    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value) -> torch._C.Value:
        return g.op("Clip", input, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))

    @staticmethod
    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in

class _FakeAQuantiser(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  x:   torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
        x = x / (step * scale) # quantized 
        x = torch.clip(x, clip_lo, clip_hi ) # clipping to bw
        x = torch.round(x)
        
        x = x * (step * scale)
        return x 
    @staticmethod
    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in