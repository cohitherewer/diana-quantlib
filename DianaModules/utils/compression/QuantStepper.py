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
        assert (steps != 0 , "Please use a non-zero step") 
        self.module = module    
        qrange = resolve_qrangespec(target_quant)
        self.target_zero, self.target_n_levels, _, _ = create_qhparams(qrange)
        qrange = resolve_qrangespec(initial_quant)
        initial_zero, initial_n_levels  , _ , _ = create_qhparams(qrange) 

        self.n_levels_step = math.ceil((initial_n_levels - self.target_n_levels )/ steps )
        if initial_n_levels <= self.target_n_levels: 
            raise RuntimeError("The QuantDownStepper module can only be used to step down quantization. Not step up!")
        
        #scale = use this get_scale(self.min_float, self.max_float, self.zero, self.n_levels, self.step) 
        #    self.scale.data.copy_(scale)
        #self._set_clipping_bounds()
        self.module_type = module_type
        self.current_step = steps 
        
        assert (initial_n_levels >= steps >= 1, "Please decrease the steps and also ensure the step count is more than or equal to 1. Else just use regular quantization")  

    def step(self) -> None :  
        if (self.current_step != 0 ): 
            map(self.step_module, filter(lambda _, m: isinstance(m, self.module_type) ,self.module.named_modules()))
            self.current_step -= 1  
    def step_module(self, module : _QModule): 
        #redefine quantization parameters  
        n_levels = -self.n_levels_step + module.n_levels
        n_levels = torch.clamp_min(n_levels, min= self.target_n_levels ) 
        factor = n_levels/module.n_levels
        zero = torch.floor(factor*module.zero)    
        zero = torch.clamp_max(zero,min=self.target_zero )  
        module.n_levels =  torch.tile(n_levels,  self.n_levels.shape) 
        module.zero =  torch.tile(zero,  self.zero.shape) 
        #redefine scale  
        scale = get_scale(module.min_float, module.max_float, module.zero, module.n_levels, module.step) 
        module.scale.data.copy_(scale)
        module._set_clipping_bounds()  

    
