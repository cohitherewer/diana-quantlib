from typing import List
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from DianaModules.utils.grapheditors import DianaAps
from DianaModules.core.Operations import AnalogGaussianNoise
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
import torch

class AnalogNoiseEnablerFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps] = [] 
        for node in g.graph.nodes: 
            try :
                if isinstance(g.get_submodule(node.target) , AnalogGaussianNoise): 
                    aps.append(DianaAps('noise' , node )) 
            except: 
                continue 
        return aps 
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 
class AnalogNoiseEnablerApplier(Applier) : 
    def __init__(self, enable_noise = False):
        super().__init__()
        self.enable_noise =enable_noise
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        node = ap.node
        module = g.get_submodule(node.target) 
        if self.enable_noise: 
            module.enable()
        # fix eps tunnels so they aren't removed in the epstunnel remover step 
        prev_eps_tunnel : EpsTunnel= g.get_submodule([p for p in node.all_input_nodes][0].target)
        next_eps_tunnel : EpsTunnel= g.get_submodule([u for u in node.users][0].target)
        assert isinstance(prev_eps_tunnel, EpsTunnel) and isinstance(next_eps_tunnel , EpsTunnel) 
        prev_eps_tunnel.set_eps_out(torch.ones_like(prev_eps_tunnel._eps_out))
        next_eps_tunnel.set_eps_in(torch.ones_like(next_eps_tunnel._eps_in))

        return g 


class AnalogNoiseEnabler(Rewriter): #insert quantidentities between 
    def __init__(self, enable_noise = False):
       super(AnalogNoiseEnabler, self).__init__(name='AnalogNoiseEnabler', symbolic_trace_fn=quantlib_symbolic_trace,finder= AnalogNoiseEnablerFinder(), applier=AnalogNoiseEnablerApplier(enable_noise = enable_noise)) 

