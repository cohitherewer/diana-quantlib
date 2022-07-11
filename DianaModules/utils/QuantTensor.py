# Implement torch tensor wrapper  that carries with it the properties of quantized tensor instead of having to get it from nn modules

import torch 
import sys 
from typing import Union , Dict

from _TrueQuantizer import _TrueQuantize 
QuantTensorSpecs = Dict[str, str]  #scale : float=1 , zero_point : int=0 , bitwidth : int=8 , clip_lo:float =0, clip_hi:float =2**8-1 

def resolve_quanttensorspecs(specs : QuantTensorSpecs) : 
    pass 
#

HANDLED_FUNCTIONS = {} # custom for add 

import functools
def implements(torch_function):
    """Register a torch function override for QuantTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
class QuantTensor(object): 
    def __init__(self, data, specs : Union[QuantTensorSpecs,None] = None,   qop :  Union[None,torch.autograd.Function ] = None, **kwargs): 
        if not qop: 
            self.qop = _TrueQuantize.apply 
        self._t = torch.as_tensor(data, **kwargs)

        self.is_quantized = False 
        #default config 
        self.scale = 1 
        self.bitwidth = 32
        self.clip_lo =  sys.float_info.min 
        self.clip_hi = sys.float_info.max 
        self.step = 1 
    def tensor(self): 
        return self._t    
    def __repr__(self) -> str:
        return "{}".format(self.tensor())
    @property 
    def data(self): 
        return self._t.data
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, '_t') else a for a in args]
        specs = tuple(a._specs  for a in args if hasattr(a, '_specs'))
        assert len(specs) > 0

        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, QuantTensor))
            for t in types
        ):
            args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
            return QuantTensor(func(*args, **kwargs), specs=specs[0])
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    @classmethod
    def ensure_tensor(data):
        if isinstance(data, QuantTensor):
            return data.tensor()
        return torch.as_tensor(data)

    @implements(torch.mul)
    def mul(input, other):
        try:
            if input._N == other._N:
                specs_i = input.specs()
                specs_o = other.specs()
                return QuantTensor(input.tensor * other.tensor, )
            else:
                raise ValueError("Shape mismatch!")
        except AttributeError:
            return torch.mul(QuantTensor.ensure_tensor(input), QuantTensor.ensure_tensor(other)) 
    @implements(torch.add)
    def add(input, other): ## assume 32-bit output 
        try:
            if input._N == other._N:
                specs_i = input.specs()
                specs_o = other.specs()
                return QuantTensor(input.tensor + other.tensor, )
            else:
                raise ValueError("Shape mismatch!")
        except AttributeError:
            return torch.add(QuantTensor.ensure_tensor(input), QuantTensor.ensure_tensor(other)) 


    def quantize_tensor(self): 
        pass 

    pass 
