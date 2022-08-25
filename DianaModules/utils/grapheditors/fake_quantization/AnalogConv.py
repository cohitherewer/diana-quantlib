
from imp import new_module
from pydoc import describe
from typing import List, Tuple
from DianaModules.core.Operations import AnalogAccumulator, AnalogConv2d, AnalogGaussianNoise, DIANAConv2d
from DianaModules.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.nametomodule.nametomodule import NameToModule
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
#  Analog conv should be placed before interposer
class AnalogConvFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps]
        aps = [] 
        #name_to_module #get names and modules with activations 
        # get modules first 
        names = []
        for name ,_ in g.named_modules(): 
            if(isinstance(g.get_submodule(name) , DIANAConv2d) and g.get_submodule(name).is_analog): 
                names.append(name)
            
        for node in g.graph.nodes: 
            if node.target in names : 
                aps.append(DianaAps('conv' ,node)) 
        
        return aps 
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 

class AnalogConvApplier(Applier) : 
    def __init__(self, description : Tuple[str , ...]): 
        self.spec = description
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        #  add  gaussian noise  ,accumulator &
        node = ap.node 
        conv_module =g.get_submodule(node.target)
        accumulator_node = None
        users = [u for u in node.users] 
        
        accum_target = id_ 
        accumulator = AnalogAccumulator() 
        noise_target = id_ + f'[{str(self._counter)}]'
        noise_module = AnalogGaussianNoise(signed=True , bitwidth=6) 
        g.add_submodule(accum_target , accumulator)
        g.add_submodule(noise_target, noise_module)
        # add noise 
        with g.graph.inserting_after(node): 
            noise_node = g.graph.call_module(noise_target , args=(node, ))  
        with g.graph.inserting_after(noise_node) : 
            accumulator_node = g.graph.call_module(accum_target , args=(noise_node,))
        for u in users: 
            u.replace_input_with(node ,accumulator_node )
        # replace DIANAConv2d with AnalogConv2D
     
        qgranularity = self.spec[0]
        qrange = self.spec[1]
        qhparaminitstrat = self.spec[2]
        analog_module = AnalogConv2d(qrangespec=qrange , qgranularityspec=qgranularity , qhparamsinitstrategyspec=qhparaminitstrat , in_channels=conv_module.in_channels , out_channels=conv_module.out_channels , kernel_size=conv_module.kernel_size , stride = conv_module.stride , padding=conv_module.padding , dilation=conv_module.dilation) #initialize the analog conv2d operation 
        #replace it 
        name_to_module = NameToModule(g.named_modules())
        name_to_module[node.target] = analog_module
        path_to_parent, child = NameToModule.split_path_to_target(node.target)
        setattr(name_to_module[path_to_parent], child, analog_module)  # https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L44

        return g

class AnalogConvMapper(Rewriter) : 
    def __init__(self ,analogconvdescriptionspec):
        finder = AnalogConvFinder() 
        applier = AnalogConvApplier(description=analogconvdescriptionspec) 
        super().__init__("AnalogConvMapper", quantlib_symbolic_trace, finder, applier)