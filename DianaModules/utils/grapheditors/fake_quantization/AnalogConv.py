
from imp import new_module
from typing import List
from DianaModules.core.Operations import Accumulator, AnalogConv2d, DIANAConv2d
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
    def __init__(self):
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        #  add accumulator         
        node = ap.node 
        accumulator_node = None
        users = [u for u in node.users] 
        accum_target = id_ 
        accumulator = Accumulator() 
        g.add_submodule(accum_target , accumulator)
        with g.graph.inserting_after(node) : 
            accumulator_node = g.graph.call_module(accumulator , args=(ap.node,))
        for u in users: 
            u.replace_input_with(node ,accumulator_node )
        # replace DIANAConv2d with AnalogConv2D
        prev_module = g.get_submodule(node.target) 

        analog_module = AnalogConv2d() #initialize the analog conv2d operation 
        #replace it 
        name_to_module = NameToModule(g.named_modules())
        name_to_module[node.target] = analog_module
        path_to_parent, child = NameToModule.split_path_to_target(node.target)
        setattr(name_to_module[path_to_parent], child, analog_module)  # https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L44

        return g

class AnalogConvMapper(Rewriter) : 
    def __init__(self, name: str):
        finder = AnalogConvFinder() 
        applier = AnalogConvApplier() 
        super().__init__(name, quantlib_symbolic_trace, finder, applier)