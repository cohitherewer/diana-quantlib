# Placed before eps tunnel inserters 

# step 3 : multiply max float and min float by BN alpha and add beta to the bias and redefine params 

from typing import List, Union
import torch 
from torch import nn
from DianaModules.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter 
import torch.fx as fx
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.pattern import NNSequentialPattern
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from DianaModules.core.Operations import AnalogConv2d, DIANAConv2d, DIANALinear, DIANAIdentity
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class

checker = (lambda m : True if type(m) != AnalogConv2d else False, )

_QOP_KWARGS = {'qrangespec': {'bitwidth': 8, 'signed': True}, 'qgranularityspec': 'per-array', 'qhparamsinitstrategyspec': 'meanstd'}

roles = Roles([
    ('linear',  Candidates([
        ('Conv', NNModuleDescription(class_=DIANAConv2d, kwargs=_QOP_KWARGS | {'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'bias': True}, checkers=checker)),
        ('FC', NNModuleDescription(class_=nn.Linear,kwargs={'in_features': 1, 'out_features': 1, 'bias': True})),
    ])),
    ('interposer', Candidates([
        ('DianaInterposer', NNModuleDescription(class_=DIANAIdentity, kwargs=_QOP_KWARGS)),
    ])),
    ('bn', Candidates([
        ('BN2d', NNModuleDescription(class_=nn.BatchNorm2d, kwargs={'num_features': 1})),
    ])),
])
screenplays = roles.all_screenplays
class BNFolderApplier(NNModuleApplier): 
    def __init__(self, pattern : NNSequentialPattern):
        super(BNFolderApplier, self).__init__(pattern=pattern)
    def _apply(self, g: fx.graph_module.GraphModule, ap: NodesMap, id_: str) -> fx.graph_module.GraphModule:
        #step 1: identify layers
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_linear  = name_to_match_node['linear']
        node_bn = name_to_match_node['bn']

        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_linear : Union[DIANAConv2d, DIANALinear] = name_to_match_module['linear']
        module_bn  = name_to_match_module['bn']
        # step 2 : absorb the BN into weights and bias 
        mi      = module_bn.running_mean
        gamma   = module_bn.weight 
        sigma   = torch.sqrt(module_bn.running_var + module_bn.eps) 
        beta    = module_bn.bias 
        scale = gamma /sigma 
        shift = (-mi * gamma + beta * sigma) / sigma 
        
        shape =  tuple(1 if i != 0 else mi.numel() for i, _ in enumerate(range(0, len(module_linear.weight.shape))))
        scale = scale.reshape(shape) 
        module_linear.weight.data = module_linear.weight.data * scale 
        if module_linear.bias is not None: 
            module_linear.bias.data = module_linear.bias.data  * scale  + shift
        else: 
            module_linear.bias = nn.Parameter(torch.empty(module_linear.out_channels, device=module_linear.weight.device, dtype=module_linear.weight.dtype)) 
            module_linear.bias.data = shift.clone().detach()
        bn_users = [u for u in node_bn.users]  
        for user in bn_users: 
            user.replace_input_with(node_bn, node_linear) 
        g.delete_submodule(node_bn.target) 
        g.graph.erase_node(node_bn)

        #step 3: requantize the original modules , if it's per-outchannel_weights then scale the max min float values and reinitialize quant hyper parameters. if it's per-array, then set _is_quantised to false and take care of it after the conversion 
        #if module_linear.scale.numel() >= 1: #per _outchannel weights 
        #    module_linear.max_float *= scale 
        #    module_linear.min_float *= scale 
        #    module_linear.redefine_qhparams({'bitwidth': 8 , 'signed': True}) # digital core qrange spec
        #else : 
        module_linear._is_quantised &= False # set quantization to false 

        return g
class BNFolder(ComposedEditor):   
    def __init__(self):
        # generate rewriters for Dianaconv2d->BN and DIANALinear>BN 
        admissible_screenplays = list(roles.all_screenplays)
        editors =[]
        for name, pattern in generate_named_patterns(roles, admissible_screenplays):
            class_name = name + 'DianaIntegeriser'
            
            class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, BNFolderApplier)
            editors.append(class_())
        super().__init__(editors)
