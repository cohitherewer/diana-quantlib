# find  pattern analog conv epstunnel identity epstunnel accumulator epstunnel 
# replace that with  DIANAConv2D and epstunnel (absorb the scaling into the dianaconv weights or the epstunnel after)  copy the weight values and biases 
from typing import List
import torch 
from torch import nn 
from DianaModules.core.Operations import AnalogAccumulator, AnalogConv2d
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
        ('QIdentity', NNModuleDescription(class_=nn.Identity, kwargs={})),
    ])),
     ('eps_identity_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
    ('accumulator', Candidates([
        ('Accumulator', NNModuleDescription(class_=AnalogAccumulator, kwargs={})),
    ])),
    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
])

class AnalogConvIntegrizerApplier(Applier) : 
    def __init__(self):
        super().__init__()
    @classmethod 
    def create_torch_module(qconv : AnalogConv2d, ) :  
        assert isinstance(qconv, AnalogConv2d)  
        new_module = nn.Conv2d(in_channels=qconv.in_channels,
                                out_channels=qconv.out_channels,
                                kernel_size=qconv.kernel_size,
                                stride=qconv.stride,
                                padding=qconv.padding,
                                dilation=qconv.dilation,
                                groups=qconv.groups,
                                bias=False)
        new_module.weight.data = torch.round(qconv.weight.data.clone().detach()  / qconv.scale.data.clone().detach()) # fake quantized / scale = true quantized
        new_module.register_buffer("is_analog" , torch.Tensor([True])) 
        new_module.register_buffer("gain", torch.zeros(qconv.gain.size(0)) )# tensor of gain values to be loaded into analog core 
        new_module.gain.data = qconv.gain.data.clone().detach() 
        return new_module
        

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
    
         # create the integerised linear operation
        new_target = id_
        new_module = AnalogConvIntegrizerApplier.create_torch_module(module_conv)
  
        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_conv_out.replace_input_with(node_conv, new_node)
        acc_users = [u for u in node_accumulator.users] 
        for u in acc_users: 
            u.replace_input_with(node_accumulator , node_identity_out)

         # TODO ...and delete conv and accumulator opertions , set epstunnels of eps_identity_out eps_out to 1 and eps_out eps_in to 1
        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_conv_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))
        module_identity_out.set_eps_out(torch.ones_like(module_identity_out.eps_out)) 
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in)) 

         # ...and delete the old operation
        g.delete_submodule(node_conv.target)
        g.graph.erase_node(node_conv)
        g.delete_submodule(node_accumulator.target)
        g.graph.erase_node(node_accumulator)
       
        return g

class AnalogConvIntegrizer(ComposedEditor):   
    def __init__(self):
        # generate rewriters for qconv2d and qlinear 
        admissible_screenplays = list(analog_roles.all_screenplays)

    # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors : List[Editor]
        editors =[]
        for name, pattern in generate_named_patterns(analog_roles, admissible_screenplays):
            class_name = name + 'DianaAnalogIntegeriser'
            class_ = get_rewriter_class(class_name, pattern, LinearOpIntegeriserMatcher, AnalogConvIntegrizerApplier)
            editors.append(class_())
        super().__init__(editors)
    pass 
#endregion
   