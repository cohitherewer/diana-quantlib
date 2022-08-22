from typing import List
import torch 
from torch import nn
from quantlib.editing.editing.editors.base.editor import Editor

from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.pattern.base.pattern import NNModulePattern 
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
import torch.fx as fx
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.fake2true.integerisation.linearopintegeriser.finder import LinearOpIntegeriserMatcher
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.nametomodule.nametomodule import NameToModule
#region LinearOpIntegrizer

# Integrizer for qlinear and dianaconv that's only digtial - 8 bit 



SUPPORTED_LINEAR_FPMODULES = (nn.Linear , nn.Conv2d) 

class DianaLinearOpIntegrizerApplier(NNModuleApplier): 
    def __init__(self, pattern: NNModulePattern):
        super().__init__(pattern)
    
    @staticmethod 
    def create_torch_module(qlinear : _QModule): 
        """Return an ``nn.Module`` implementing a linear operation with
        integerised parameters.
        """

        if not isinstance(qlinear, SUPPORTED_LINEAR_FPMODULES):
            raise TypeError
        if issubclass(type(qlinear), nn.Linear):
            new_module = nn.Linear(in_features=qlinear.in_features,
                                out_features=qlinear.out_features,
                                bias=(qlinear.bias is not None))
        elif issubclass(type(qlinear), nn.Conv2d):
                new_module = nn.Conv2d(in_channels=qlinear.in_channels,
                                out_channels=qlinear.out_channels,
                                kernel_size=qlinear.kernel_size,
                                stride=qlinear.stride,
                                padding=qlinear.padding,
                                dilation=qlinear.dilation,
                                groups=qlinear.groups,
                                bias=(qlinear.bias is not None))
        
        else:
            raise RuntimeError
        new_module.register_buffer("is_analog" , torch.Tensor([qlinear.bias is not None]) ) # register buffer inside new_module to annotate the model correctly in the onnx export stage 
 
        iweight = torch.round(qlinear.qweight.data.clone().detach() / qlinear.scale.data.clone().detach())  # integerised parameters
        new_module.weight.data = iweight
        if qlinear.bias is not None: 
            new_module.bias.data = qlinear.bias.data.clone().detach().round() # we don't want to detach the original tensor + integrization 
            new_module.bias.type(torch.int32)
        
        return new_module 
    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in  = name_to_match_node['eps_in']
        node_linear  = name_to_match_node['linear']
        node_eps_out = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in  = name_to_match_module['eps_in']
        module_linear  = name_to_match_module['linear']
        module_eps_out = name_to_match_module['eps_out']
        if not issubclass(type(module_linear) , _QModule): 
            # already edited. node might be returned multiple times due to nn.sequential 
            return g 


        # create the integerised linear operation
        new_target = id_
        new_module = DianaLinearOpIntegrizerApplier.create_torch_module(module_linear)
  
        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_linear, new_node)
       
        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))
         # ...and delete the old operation
        g.delete_submodule(node_linear.target)
        g.graph.erase_node(node_linear)
       
        name_to_module = NameToModule(g.named_modules())
        name_to_module[node_linear.target] = new_module
        path_to_parent, child = NameToModule.split_path_to_target(node_linear.target)
        setattr(name_to_module[path_to_parent], child, new_module)  # this is needed for nn sequentials 
        
        

        return g

##################### DEFINING VARS #########################
checker =( lambda m: True , ) 
analog_checker =( lambda m: False if m.is_anlog else True  , ) 
di_roles = Roles([

    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),

    ('linear', Candidates([
        ('QLinear', NNModuleDescription(class_=nn.Linear, kwargs={'in_features': 1, 'out_features': 1, 'bias': True}, checkers=checker)) , 
        ('QConv2d', NNModuleDescription(class_=nn.Conv2d , kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'bias': True}, checkers=analog_checker))
    ])),

    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
])
   
class DianaLinearOpIntegrizer(ComposedEditor):   
    def __init__(self):
        # generate rewriters for qconv2d and qlinear 
        admissible_screenplays = list(di_roles.all_screenplays)

    # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors : List[Editor]
        editors =[]
        for name, pattern in generate_named_patterns(di_roles, admissible_screenplays):
            class_name = name + 'DianaIntegeriser'
            class_ = get_rewriter_class(class_name, pattern, LinearOpIntegeriserMatcher, DianaLinearOpIntegrizerApplier)
            editors.append(class_())
        super().__init__(editors)
    pass 
#endregion
   