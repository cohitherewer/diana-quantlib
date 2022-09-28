import torch 
import torch.fx as fx 
from torch import nn
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.editing.editing.editors.base.editor import Editor

class LinearLayerQuantizer(Editor): 
    def __init__(self) -> None:
        super().__init__()
    def apply(self, g: fx.graph_module.GraphModule, *args, **kwargs) -> fx.graph_module.GraphModule:
        x = kwargs['input'] 
       
        for _ , mod in g.named_modules(): 
            if isinstance(mod , _QModule) and mod._is_quantised & True == False: 
                mod.start_observing()
        _ = g(x)
        for _ , mod in g.named_modules(): 
            if isinstance(mod , _QModule) and mod._is_quantised & True == False: 
                mod.stop_observing()
                mod.scale = torch.exp2(torch.round(torch.log2(mod.scale)))
        return g