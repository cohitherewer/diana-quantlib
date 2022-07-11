import torch



class _FakeDQuantiser(torch.autograd.Function): # symmetric quantisation 
    
    def forward(ctx,  x:   torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
        x = x / (step * scale)
        x = torch.clip(x, clip_lo, clip_hi )
        x = torch.floor(x)
        x = x * (step * scale)
        return x 

    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in

class _FakeAQuantiser(torch.autograd.Function):
    
    def forward(ctx,  x:   torch.Tensor,
                   clip_lo: torch.Tensor,
                   clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
        passx = torch.clip(x, clip_lo, clip_hi )
     #   x = x - clip_lo #uncomment for unsymmetric mode 
        x = x / (step * scale)
       
        x = torch.round(x)
         
        x = x * (step * scale)
        return x 

    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in