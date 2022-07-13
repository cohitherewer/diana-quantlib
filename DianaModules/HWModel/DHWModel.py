import numpy

# bn_w  quantized batchnorm weight
# bn_ws batchnorm weight scale
# bn_b batchnorm bias 
# bn_bs batchnorm bias scale 
# r_s scale of tensor 
# q_s quantization scale  
         

class SIMDModel:
    @classmethod
    def _bn(cls, layer, in_fmap): # batch norm 
        out_fmap = numpy.zeros(in_fmap.shape)
        for c in range(in_fmap.shape[1]):
            out_fmap[0][c] = (layer.bn_w[c][0]*(2**layer.bn_ws))*in_fmap[0][c] + (layer.bn_b[c]*2**layer.bn_bs)
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
        min_val = -2**(bw-1)+1 # TODO weird clipping bound. Ask about this 
        clip_min_mask = (in_fmap<min_val).astype(int)
        clip_max_mask = (in_fmap>max_val).astype(int)
        clip_min_mask_n = numpy.logical_xor(clip_min_mask, 1).astype(int)
        clip_max_mask_n = numpy.logical_xor(clip_max_mask, 1).astype(int)
        min_filler = numpy.ones(in_fmap.shape)*clip_min_mask*min_val
        max_filler = numpy.ones(in_fmap.shape)*clip_max_mask*max_val
        out_fmap = (in_fmap*clip_min_mask_n*clip_max_mask_n)+min_filler+max_filler
        return out_fmap
    @classmethod
    def _quant(cls, layer, in_fmap, ds=False): # quantize 
        out_fmap = numpy.floor(in_fmap/2**layer.q_s)
        out_fmap = cls._clip(out_fmap,7)
        return out_fmap
    @classmethod
    def fp(cls, layer, in_fmap0, in_fmap1=None, ds=False): # forward pass 
        x = cls._bn(layer, in_fmap0)
        if type(in_fmap1) == numpy.ndarray:
            x = cls._rs(layer, x, in_fmap1)
        if ds==False:
            x = cls._relu(x)
        x = cls._quant(layer, x)
        return x
    @classmethod
    def fp_dbg(cls, layer, in_fmap0, in_fmap1=None, ds=False):
        out = []
        out.append(cls._bn(layer, in_fmap0))
        i=0
        if type(in_fmap1) == numpy.ndarray:
            out.append(cls._rs(layer, out[i], in_fmap1))
            i+=1
        if ds==False:
            out.append(cls._relu(out[i]))
            i+=1
        out.append(cls._quant(layer, out[i]))
        return out