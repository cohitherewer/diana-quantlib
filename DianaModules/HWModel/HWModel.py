from typing import Union
import numpy as np 

# bn_w  quantized batchnorm weight
# bn_ws batchnorm weight scale
# bn_b batchnorm bias 
# bn_bs batchnorm bias scale 
# r_s scale of tensor 
# q_s quantization scale  
         

class SIMDModel:
    '''
    SIMD allowed sequence of operations:
    - Partial sum
    - ReLU + Quant
    - BN + Quant
    - Res
    - Res + ReLU + Quant
    - Partial sum + BN + ReLU + Quant
    - BN + ReLU + Quant
    - BN + Res + ReLU + Quant
    - BN + Res + ReLU + Quant + Pool
    - BN + ReLU + Quant + Pool
    - NOP
    '''
    @classmethod
    def _bn(cls, layer, in_fmap , row =0 ): # batch norm 
        out_fmap = np.zeros(in_fmap.shape)
        for c in range(in_fmap.shape[1]):
            out_fmap[row][c] = (layer.bn_w[c][row]*(2**layer.bn_ws))*in_fmap[row][c] + (layer.bn_b[c]*2**layer.bn_bs)
        return out_fmap
    @classmethod
    def _rs(cls, layer, in_fmap0, in_fmap1): #residual add 
        out_fmap = in_fmap0 + in_fmap1*(2**layer.r_s)
        return out_fmap
    @classmethod
    def _relu(cls, in_fmap): 
        out_fmap = in_fmap*(in_fmap>0).astype(int)
        return out_fmap 
    @classmethod
    def _clip(cls, in_fmap, bw):  
        max_val = 2**(bw-1)-1
        min_val = -2**(bw-1)+1
        clip_min_mask = (in_fmap<min_val).astype(int)
        clip_max_mask = (in_fmap>max_val).astype(int)
        clip_min_mask_n = np.logical_xor(clip_min_mask, 1).astype(int)
        clip_max_mask_n = np.logical_xor(clip_max_mask, 1).astype(int)
        min_filler = np.ones(in_fmap.shape)*clip_min_mask*min_val
        max_filler = np.ones(in_fmap.shape)*clip_max_mask*max_val
        out_fmap = (in_fmap*clip_min_mask_n*clip_max_mask_n)+min_filler+max_filler
        return out_fmap
    @classmethod
    def _quant(cls, layer, in_fmap, ds=False, bitwidth= 7 ): # quantize 
        out_fmap = np.floor(in_fmap/2**layer.q_s)
        out_fmap = SIMDModel._clip(out_fmap,bitwidth)
        return out_fmap
    @classmethod
    def fp(cls, layer, in_fmap0, in_fmap1=None, ds=False): # forward pass 
        x = SIMDModel._bn(layer, in_fmap0)
        if type(in_fmap1) == np.ndarray:
            x = SIMDModel._rs(layer, x, in_fmap1)
        if ds==False:
            x = SIMDModel._relu(x)
        x = SIMDModel._quant(layer, x)
        return x
    @classmethod
    def simd_op(cls, layer , in_fmap0 : np.ndarray, in_fmap1:Union[np.ndarray, None] =None): # forward pass 

        x = SIMDModel._bn_f(layer, in_fmap0)
        if in_fmap1 is not None:
            x = SIMDModel._rs(layer, x, in_fmap1)
        if layer.relu==True:
            x = SIMDModel._relu(x)
        x = SIMDModel._quant(layer, x,bitwidth=layer.bitwidth_clip)
        return x  
    @classmethod 
    def _bn_f(cls, layer, in_fmap ): # batch norm 
        out_fmap = np.zeros(in_fmap.shape)
        for batch in range(in_fmap.shape[0]):  
            for channel in range(in_fmap.shape[1]): 
                for row in range(in_fmap.shape[2]): 
                    for c in range(in_fmap.shape[3]):
                        out_fmap[batch][channel][row][c] = (layer.bn_w[0][channel][0][0].item()*(2**layer.bn_ws))*in_fmap[batch][channel][row][c].item() + (layer.bn_b[0][channel][0][0].item()*2**layer.bn_bs)
        return out_fmap
    @classmethod
    def fp_dbg(cls, layer, in_fmap0, in_fmap1=None, ds=False):
        out = []
        out.append(SIMDModel._bn(layer, in_fmap0))
        i=0
        if type(in_fmap1) == np.ndarray:
            out.append(SIMDModel._rs(layer, out[i], in_fmap1))
            i+=1
        if ds==False:
            out.append(SIMDModel._relu(out[i]))
            i+=1
        out.append(SIMDModel._quant(layer, out[i]))
        return out
