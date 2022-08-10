import torch

class _FakeDQuantiser(torch.autograd.Function): # symmetric quantisation 
    @staticmethod
    def forward(ctx,  x:   torch.Tensor,
                   bw_clip_lo: torch.Tensor,
                   bw_clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor) : 
   
        
        
        x = x / (step * scale) # quantized 
        x = torch.clip(x, bw_clip_lo, bw_clip_hi ) # clipping to bw
        x = torch.floor(x)
        
        x = x * (step * scale)
        return x 

    @staticmethod
    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in , None, None, None, None

class _FakeAQuantiser(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  x:   torch.Tensor,
                   bw_clip_lo: torch.Tensor,
                   bw_clip_hi: torch.Tensor,
                   step:    torch.Tensor,
                   scale:   torch.Tensor , truncate : bool = False ) : 
        x = x / (step * scale) # quantized 
        x = torch.clip(x, bw_clip_lo, bw_clip_hi ) # clipping to bw
        x = torch.round(x)  if not truncate else  torch.floor(x * 2** 6 ) / 2**6 
        
        x = x * (step * scale)
        return x 
    @staticmethod
    def backward(   ctx, g_in ) : # straight through estimator 
        return g_in , None , None , None , None