import torch


class _TrueQuantize(torch.autograd.Function):
    
    def forward(ctx,  x:   torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
        passx = torch.clip(x, clip_lo, clip_hi )
     #   x = x - clip_lo #uncomment for unsymmetric mode 
        x = x / (step * scale)
       
        x = torch.round(x)
         
    
        return x 

    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in