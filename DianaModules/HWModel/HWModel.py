import numpy

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
        min_val = -2**(bw-1)+1
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

class DCore :
    @classmethod 
    def create_conv(i_layer, layer_node, dory_node, network_dir, input=None, weight=None, batchnorm_params=None):
        x = input if input is not None else create_input(layer_node)
        x_save = x.flatten()
        if input is None:
            np.savetxt(os.path.join(network_dir, 'input.txt'), x_save, delimiter=',', fmt='%d')

        w = weight if weight is not None else create_weight(layer_node)
        layer_node.constant_names.append('weights')
        layer_node.weights = {
            'value': w.numpy(),
            'layout': 'CoutCinK'
        }
        b = create_bias(layer_node)
        layer_node.constant_names.append('bias')
        layer_node.bias = {
            'value': b.numpy(),
            'layout': ''
        }

        y = F.conv2d(input=x, weight=w, bias=b, stride=layer_node.strides, padding=layer_node.pads[0], groups=layer_node.group)
        y_type = torch.int32
        y = y.type(y_type)
        y_signed = layer_node.output_activation_type == 'int'

        dory_node.constant_names.append('outmul')
        dory_node.outmul = {
            'value': 1,
            'layout': ''
        }

        dory_node.constant_names.append('outshift')
        dory_node.outshift = {
            'value': calculate_shift(y, dory_node.output_activation_bits, y_signed),
            'layout': ''
        }
        y = y >> dory_node.outshift['value']
        y = clip(y, dory_node.output_activation_bits, y_signed)

        y_save = copy.deepcopy(y.flatten())
        y_save = y_save.reshape(int(y_save.shape[0]/4), 4)
        y_save1 = copy.deepcopy(y_save)
        y_save[:,0] = y_save1[:,3] 
        y_save[:,1] = y_save1[:,2] 
        y_save[:,2] = y_save1[:,1] 
        y_save[:,3] = y_save1[:,0] 
        y_save = y_save.flatten().numpy()
        np.savetxt(os.path.join(network_dir, f'out_layer{i_layer}.txt'), y_save, delimiter=',', fmt='%d')
        return y
