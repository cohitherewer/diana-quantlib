# find  pattern analog conv epstunnel identity epstunnel accumulator epstunnel 
# replace that with  DIANAConv2D and epstunnel (absorb the scaling into the dianaconv weights or the epstunnel after)  copy the weight values and biases 
from typing import List
import torch 
from torch import nn 
from DianaModules.core.Operations import Accumulator, AnalogConv2d
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.editing.fake2true.integerisation.linearopintegeriser.finder import LinearOpIntegeriserMatcher
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
import torch.fx as fx

Analogchecker =( lambda m: True  ) 
analog_roles  = Roles([


    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
    ('conv', Candidates([
        ('QConv2d', NNModuleDescription(class_=AnalogConv2d, kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'bias': True}))
    ])),

    ('eps_conv_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
     ('identity', Candidates([
        ('QIdentity', NNModuleDescription(class_=nn.Identity)),
    ])),
     ('eps_identity_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
    ('accumulator', Candidates([
        ('Accumulator', NNModuleDescription(class_=Accumulator)),
    ])),
    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
])

class AnalogConvIntegrizerApplier(Applier) : 
    def __init__(self):
        super().__init__()
    @classmethod 
    def create_torch_module(module : AnalogConv2d, *args) :  
        pass 

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in  = name_to_match_node['eps_in']
        node_conv  = name_to_match_node['conv']
        node_conv_out = name_to_match_node['eps_conv_out']
        node_identity = name_to_match_node['identity']
        node_identity_out= name_to_match_node['eps_identity_out'] # should have same eps in as node_conv_out epsout
        node_accumulator  = name_to_match_node['accumulator']
        node_eps_out = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in  = name_to_match_node['eps_in']
        module_conv  = name_to_match_node['conv']
        module_conv_out = name_to_match_node['eps_conv_out']
        module_identity = name_to_match_node['identity']
        module_identity_out= name_to_match_node['eps_identity_out'] # should have same eps in as node_conv_out epsout
        module_accumulator  = name_to_match_node['accumulator']
        module_eps_out = name_to_match_node['eps_out']
        if not issubclass(type(module_conv) , _QModule): 
            # already edited. node might be returned multiple times due to nn.sequential 
            return g 
         # create the integerised linear operation
        new_target = id_
        new_module = AnalogConvIntegrizerApplier.create_torch_module(module_conv)
  
        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_accumulator, new_node)
       
        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))
         # TODO ...and delete all the old operations
       
        return g

class AnalogConvIntegrizer(ComposedEditor):   
    def __init__(self):
        # generate rewriters for qconv2d and qlinear 
        admissible_screenplays = list(analog_roles.all_screenplays)

    # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors : List[Editor]
        editors =[]
        for name, pattern in generate_named_patterns(analog_roles, admissible_screenplays):
            class_name = name + 'DianaIntegeriser'
            class_ = get_rewriter_class(class_name, pattern, LinearOpIntegeriserMatcher, DianaLinearOpIntegrizerApplier)
            editors.append(class_())
        super().__init__(editors)
    pass 
#endregion
   