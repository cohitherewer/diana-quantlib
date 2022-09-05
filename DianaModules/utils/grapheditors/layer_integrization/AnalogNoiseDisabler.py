from typing import List
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from DianaModules.utils.grapheditors import DianaAps
from DianaModules.core.Operations import AnalogGaussianNoise
from quantlib.editing.graphs.fx import quantlib_symbolic_trace

class AnalogNoiseDisablerFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps] = [] 
        for node in g.graph.nodes: 
            try: 
                if isinstance(g.get_submodule(node.target) , AnalogGaussianNoise): 
                    aps.append(DianaAps('noise' , node )) 
            except: 
                continue
        return aps 
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 
class AnalogNoiseDisablerApplier(Applier) : 

    def __init__(self):
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        
        node = ap.node
        noise_module = g .get_submodule(node.target) 
        noise_module.disable() 
        #predecessor = [p for p in node.all_input_nodes] 
        #users = [ u for u in node.users] 
        #assert len(predecessor) == 1
        #print("Analog Noise predecessors , ", *predecessor)
        
        #print("Analog Noise users, ", *users)
        #print("analog noise removed ")
        #for u in users: 
        #    u.replace_input_with(node , predecessor[0]) 

        #g.delete_submodule(node.target) 
        #g.graph.erase_node(node)
        return g 


class AnalogNoiseDisabler(Rewriter): #insert quantidentities between 
    def __init__(self):
       super(AnalogNoiseDisabler, self).__init__(name='AnalogNoiseRemover', symbolic_trace_fn=quantlib_symbolic_trace,finder= AnalogNoiseDisablerFinder(), applier=AnalogNoiseDisablerApplier()) 

