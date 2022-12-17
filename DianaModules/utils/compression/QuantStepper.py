import math
import torch
from typing import Any, Tuple, Union
from DianaModules.core.Operations import AnalogConv2d
from DianaModules.utils.BaseModules import DianaModule
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams, get_scale
from quantlib.algorithms.qbase.qrange.qrange import QRangeSpecType, resolve_qrangespec
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
# ---------------------------------------------------------------------------- #
#                 Wrapper Class that steps every training iteration            #
#                Quantization Stepper For _QModule Layers(only steps down)     # 
#                    Example(steps = 2): 8b -> 5b -> 3b(ternary)               #
#       Only works with 1 class type.. if you need more create more instances  #
# ---------------------------------------------------------------------------- #   
class QuantDownStepper: 
    def __init__(self, module : DianaModule, steps : int, initial_quant: QRangeSpecType, target_quant: QRangeSpecType, module_type= AnalogConv2d) -> None: 
        assert steps != 0 , "Please use a non-zero step"
        self.module = module    
        qrange = resolve_qrangespec(target_quant)
        self.target_quant = qrange
        self.target_zero, self.target_n_levels, _, _ = create_qhparams(qrange)
        qrange = resolve_qrangespec(initial_quant)
        initial_zero, initial_n_levels  , _ , _ = create_qhparams(qrange) 
        self.n_levels_step = torch.ceil((torch.round(torch.log2(initial_n_levels)) - torch.round(torch.log2(self.target_n_levels)))/ steps  ) 
        print(self.n_levels_step)
        if initial_n_levels <= self.target_n_levels: 
            raise RuntimeError("The QuantDownStepper module can only be used to step down quantization. Not step up!")
        
        #scale = use this get_scale(self.min_float, self.max_float, self.zero, self.n_levels, self.step) 
        #    self.scale.data.copy_(scale)
        #self._set_clipping_bounds()
        self.module_type = module_type
        self.current_step = steps 
        
        assert initial_n_levels >= steps >= 1, "Please decrease the steps and also ensure the step count is more than or equal to 1. Else just use regular quantization"

    def step(self) -> None :  
        if (self.current_step != 0 ): 
            print(self.current_step) 
            self.current_step -= 1 
            for _,module in list(filter(lambda m: True if isinstance(m[1], self.module_type) else False ,self.module.named_modules())): 
                self.step_module(module)
             
    def step_module(self, module : _QModule): 
        #redefine quantization parameters  
        n_levels = torch.round(torch.exp2(torch.log2(module.n_levels) - self.n_levels_step))   
        n_levels = torch.clip(n_levels , min=self.target_n_levels) 
        zero = torch.floor((module.n_levels-n_levels )/2) + module.zero
        zero = torch.clip(zero, max=self.target_zero) 
        
        
        if self.current_step == 0: 
            zero = self.target_zero
            n_levels = self.target_n_levels
        print(f"ZERO:  Prev: {module.zero}  , new: {zero} ") 
        print(f"n_levels:  Prev: {module.n_levels}  , new: {n_levels} ") 
        module.n_levels =  torch.tile(n_levels,  module.n_levels.shape) 
        module.zero =  torch.tile(zero,  module.zero.shape) 
        #module.n_levels.data.copy_(n_levels)
        #module.zero.data.copy_(zero)

        #redefine scale  
        scale = get_scale(module.min_float, module.max_float, module.zero, module.n_levels, module.step) 
        module.scale.data.copy_(scale)
        module._set_clipping_bounds()  
        if module.n_levels <= 3: 
            module.bw_clip_lo = torch.tile(torch.Tensor([-1]) , module.clip_lo.shape) 
            module.bw_clip_hi = torch.tile(torch.Tensor([1]) , module.clip_hi.shape)
        else: 
            module.define_bitwidth_clipping() 
    
