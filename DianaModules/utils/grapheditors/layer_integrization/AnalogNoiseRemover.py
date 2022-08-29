from typing import List
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from DianaModules.utils.grapheditors import DianaAps
from DianaModules.core.Operations import AnalogGaussianNoise
from quantlib.editing.graphs.fx import quantlib_symbolic_trace

class AnalogNoiseRemoverFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps] = [] 
        for node in g.graph.nodes: 
            if isinstance(g.get_submodule(node.target) , AnalogGaussianNoise): 
                aps.append(DianaAps('noise' , node )) 
        return aps 
class AnalogNoiseRemoverApplier(Applier) : 
    def __init__(self):
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        node = ap.node
        predecessor = [p for p in node.all_input_nodes] 
        assert len(predecessor) == 1
        users = [ u for u in node.users] 
        for u in users: 
            u.replace_input_with(node , predecessor[0]) 
        g.delete_submodule(node.target) 
        g.graph.erase_node(node)
        return g 


class AnalogNoiseRemover(Rewriter): #insert quantidentities between 
    def __init__(self):
       super(AnalogNoiseRemover, self).__init__(name='AnalogNoiseRemover', symbolic_trace_fn=quantlib_symbolic_trace,finder= AnalogNoiseRemoverFinder(), applier=AnalogNoiseRemoverApplier()) 

