from ast import Param


import math
from attr import dataclass 
import numpy as np 
import torch


from    DianaModules.core.Operations import DQScaleBias, DIANAIdentity
from DianaModules.HWModel.HWModel import SIMDModel
from typing  import Union


#TODO ASK ABOUT NEGATIVE BOUND IN DHWMODEL 

@dataclass
class ParamLayer: 
    bn_w : Union[np.ndarray , None] =None#quantized batchnorm weights
    bn_ws : Union[np.ndarray , int] =1 #batchnorm weight scale 
    bn_b : Union[np.ndarray , None] =None #batchnorm bias 
    bn_bs : Union[np.ndarray , int] =1 #batchnorm bias scale 2**b
    r_s : int =1  #scale of tensor 2**r_s
    q_s : int =1#quantization scale  2**scale
    def print_params(self): 
        print(f'bn_w: {self.bn_w} \nbn_ws: {self.bn_ws} \nbn_b: {self.bn_b} \nbn_bs: {self.bn_bs} \nr_s: {self.r_s} \nq_s: {self.q_s} ')
import unittest
class SIMDValidation(unittest.TestCase): 
    def setUp(self):
        qrangespec = {'bitwidth': 7 , 'signed': True}
        qgranularityspec= 'per-array'
        qhparaminit =  'const'
        self.diana_bn = DQScaleBias(qrangespec , qgranularityspec, qhparaminit, in_channels=1 )  
        self.qres = DIANAIdentity(qrangespec , qgranularityspec, qhparaminit)
        self.qout= DIANAIdentity(qrangespec , qgranularityspec, qhparaminit)

        self.diana_bn.stop_observing() # _is_quantised on scales set  
        self.qres.stop_observing() 
        self.qout.stop_observing() 
        self.qres.scale = torch.tensor(2**math.floor(math.log2(self.qres.scale))) 
        self.qout.scale = torch.tensor(2**math.floor(math.log2(self.qres.scale))) 
        self.diana_bn.clip_scales() # clip to power of 2 
    def test_bn(self) : # test needs to be edited because of output of dqscale bias was edited
        
        
        
        input = torch.rand(3,3) 
        
        #quantized * scale = fp 
        #log 2 = x => 2^x = scale
  
        bn_ws = math.log2(self.diana_bn.qscale.scale.item())
        bn_bs = math.log2(self.diana_bn.qidentity.scale.item())
 
        d_out = self.diana_bn(input)# because output is usually fake quantised , I have to divide by weight scale 
        broadcasted_weights = self.diana_bn.weights.expand(input.shape)
        broadcasted_biases = self.diana_bn.biases.expand(input.shape[1])
        bn_w = (self.diana_bn.qscale(broadcasted_weights) / self.diana_bn.qscale.scale  ).detach().numpy()# quantized weights 
        bn_b = (self.diana_bn.qidentity(broadcasted_biases) / self.diana_bn.qidentity.scale ).detach().numpy()# quantized biases 
        layer = ParamLayer(bn_w = bn_w,bn_b=bn_b, bn_ws=bn_ws, bn_bs=bn_bs )

        hw_out = SIMDModel._bn( layer, input.detach().numpy())  
   
        self.assertTrue(np.allclose(d_out.detach().numpy()[0], hw_out[0], rtol=1e-05, atol=1e-08))
         
    def test_quant(self):
        q_s = math.log2(self.diana_bn.qidentity.scale.item())   
        quantized_bias = (self.diana_bn.qidentity(self.diana_bn.biases) / self.diana_bn.qidentity.scale ) # division here because of fake quantising function used 
        layer = ParamLayer(q_s=q_s) 
        out = SIMDModel._quant(layer, self.diana_bn.biases.item())
        self.assertAlmostEqual(quantized_bias.item() , out)
         

    def test_fp(self): 
        input1 = torch.rand(3,3) 
        input2 = torch.rand(3,3) 
        broadcasted_weights = self.diana_bn.weights.expand(input1.shape)
        broadcasted_biases = self.diana_bn.biases.expand(input1.shape[1])
        bn_w = (self.diana_bn.qscale(broadcasted_weights) / self.diana_bn.qscale.scale  ).detach().numpy()# quantized weights 
        bn_b = (self.diana_bn.qidentity(broadcasted_biases) / self.diana_bn.qidentity.scale ).detach().numpy()# quantized biases 
        bn_ws = math.log2(self.diana_bn.qscale.scale.item())
        bn_bs = math.log2(self.diana_bn.qidentity.scale.item())
        r_s = math.log2(self.qres.scale) 
        q_s = math.log2(self.qout.scale) 
        fmap2 = (self.qres(input2) / self.qres.scale).detach().numpy()# quantized value 
        layer = ParamLayer(bn_w = bn_w,bn_b=bn_b, bn_ws=bn_ws, bn_bs=bn_bs ,r_s=r_s , q_s =q_s  )
        hw_out = SIMDModel.fp(layer, in_fmap0= input1, in_fmap1=fmap2, ds=True)
        d_out = self.qout(self.diana_bn(input1) + self.qres(input2))  / (self.qout.scale *self.qout.step)
        
        self.assertTrue(np.allclose(d_out.detach().numpy()[0], hw_out[0], rtol=1e-05, atol=1e-08))
        pass 




if __name__ == "__main__": 

    unittest.main()