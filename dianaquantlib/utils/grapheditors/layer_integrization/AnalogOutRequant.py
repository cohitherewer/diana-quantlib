import torch 
from torch import nn
from dianaquantlib.core.Operations import AnalogAccumulator, AnalogOutIdentity

from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher
from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher


from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
import torch.fx as fx
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap

from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
import dianaquantlib.utils.Requantizers.DigitalRequant as dq
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
_EPS_KWARGS = {'eps': torch.Tensor([1.0])}

roles = Roles([

     ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),
    ('activation', Candidates([
        ('QIdentity',  NNModuleDescription(class_=AnalogOutIdentity,  kwargs={'qgranularityspec': 'per-array' , 'qhparamsinitstrategyspec':'meanstd'} ,))
    ])),

    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),
])
admissible_screenplays = list(roles.all_screenplays)


class AnalogOutRequantizerApplier(NNModuleApplier): # this will probably have to be rewritten 
    def __init__(self,
                 pattern: NNSequentialPattern,
                ):  # the integer bit-shift parameter

        super(AnalogOutRequantizerApplier, self).__init__(pattern)
    

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in     = name_to_match_node['eps_in']
        node_activation = name_to_match_node['activation']
        
        node_eps_out    = name_to_match_node['eps_out']
        
        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in     = name_to_match_module['eps_in']
        module_activation = name_to_match_module['activation']
      
        module_eps_out    = name_to_match_module['eps_out']

        
        eps_in  = module_eps_in.eps_out
        eps_out = module_eps_out.eps_in
        assert torch.all(eps_out == module_activation.scale)
        div = eps_out             / ( eps_in)
     
        new_module = dq.DigitalRequantizer( div=div, zero=module_activation.zero, n_levels=module_activation.n_levels)
        new_target = id_
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_activation, new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))

        # ...and delete the old construct
        g.delete_submodule(node_activation.target)
        g.graph.erase_node(node_activation)  # since `node_activation` is a user of `node_bn`, we must delete it first
      
        return g

# create the general-purpose `Requantiser`
class AnalogOutRequantizer(ComposedEditor):
    def __init__(self):
        namespace= {}
        for name, pattern in generate_named_patterns(roles, admissible_screenplays):
            class_name = name + 'analogout'

            class_ = get_rewriter_class(class_name, pattern, RequantiserMatcher, AnalogOutRequantizerApplier)
            namespace[class_name] = class_   
        super(AnalogOutRequantizer, self).__init__([class_() for class_ in namespace.values()])
