
from typing import List, Union

import torch

from DianaModules.core.operations import DIANAIdentity, DIANALinear, IdentityType , DIANAConv2d

from quantlib.editing.editing.editors.base.composededitor import ComposedEditor

from quantlib.editing.editing.editors.base.rewriter.applicationpoint import ApplicationPoint
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.rewriter.rewriter import NNModuleRewriter
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.fake2true import F2TConverter
from quantlib.editing.editing.fake2true.annotation import F2TAnnotator
from quantlib.editing.editing.fake2true.annotation.inputdescription import InputDescription, InputDescriptionSpecType
from quantlib.editing.editing.fake2true.epstunnels.inserter.rewriter import EpsTunnelInserter
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import EpsTunnelRemover
from quantlib.editing.editing.fake2true.epstunnels.simplifier.rewriter import EpsTunnelConstructSimplifier
from quantlib.editing.editing.fake2true.integerisation.linearopintegeriser import LinearOpIntegeriser
from quantlib.editing.editing.fake2true.integerisation.requantiser.applier import RequantiserApplier
from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher
from quantlib.editing.editing.float2fake.canonicalisation import F2FCanonicaliser

from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.retracer import QuantLibHarmonisedAddRetracer
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.rewriter import AddTreeHarmoniser

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescriptionSpecType
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.rewriter import ModuleWiseConverter
from quantlib.editing.editing.float2fake.quantisation.qdescription.qdescription import QDescriptionSpecType

import torch.fx as fx
from quantlib.editing.graphs.fx import quantlib_symbolic_trace

class DianaF2FConverter(ComposedEditor):
    """General-purpose converter to map floating-point networks into
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 qinterposerqdescriptionspec: Union[QDescriptionSpecType, None] = None):

        super(DianaF2FConverter, self).__init__([
            F2FCanonicaliser(),
            DianaF2FQuantiser(
                modulewisedescriptionspec,
                addtreeqdescriptionspec,
                addtreeforceoutputeps,
            )
        ])
# this assumes that each module has fake-quantized identity output mapping (NO need for QuantiserInterposer).
class DianaF2FQuantiser(ComposedEditor):
    """Specific Rewriter for the diana chip """

    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 qinterposerqdescriptionspec: Union[QDescriptionSpecType, None] = None):
        if qinterposerqdescriptionspec == None:
            super(DianaF2FQuantiser, self).__init__([
            QuantLibRetracer(),
            ModuleWiseConverter(modulewisedescriptionspec),
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            ),DianaF2FInterposer() 

        ]) # Add interposer here 

# Modules to look for DIANAConv2d , DIANALinear , AvgPool2d  
from torch import  nn 

class _DianaNode:
    def __init__(self, type:str, node: fx.node.Node) -> None:
        self._node = node 
        self._type = type 
    @property 
    def node(self): 
        return self._node
    @property 
    def type(self): 
        return self._type
   

class DianaAps(ApplicationPoint, _DianaNode):
    
    pass
MODULES_WITH_QUANTIZERS = [DIANAConv2d , DIANALinear , nn.AvgPool2d , nn.AdaptiveAvgPool2d]
class DianaOpQuantFinder(Finder):

    def __init__(self):
        super().__init__()
    
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        def getmodule_name(module) :
            if type(module) == DIANAConv2d: 
                return 'conv' 
            elif type(module) == DIANALinear: 
                return 'linear'
            elif type(module) == nn.AvgPool2d or type(module) == nn.AdaptiveAvgPool2d: 
                return 'pool'
            return ''
     
        aps: List[DianaAps] = []
        # get names of submodules that match the search 
        name_to_module = {}
        for name,mod in g.named_modules(): 
            if type(mod) in MODULES_WITH_QUANTIZERS:
                name_to_module[name] = mod

        for n in g.graph.nodes: 
            if n.target in name_to_module.keys(): 
                aps.append(DianaAps(getmodule_name(name_to_module[n.target]), n))
            

        return aps
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once

class DianaOpQuantApplier(Applier): 
    def __init__(self) -> None:
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
       
        type_in = IdentityType.default
        type_out = IdentityType.default
        if ap.type == 'conv' and g.get_submodule(ap.node.target).is_analog: 
            type_in = IdentityType.AIMC_IN
            type_out = IdentityType.AIMC_OUT
        qpre = DIANAIdentity({'bitwidth': 8, 'signed': True} , 'per-array', 'minmax', type_in)
        qpost = DIANAIdentity({'bitwidth': 8, 'signed': True} , 'per-array', 'minmax', type_out)
        pre_target = id_ 
        
        
        post_target = id_ + f'[{str(self._counter)}]'

        g.add_submodule(pre_target ,qpre) 
        g.add_submodule(post_target, qpost) 
            
        input_node = None
        # get input x of qpre 
        if len(ap.node.all_input_nodes) ==1: 
            input_node = ap.node.all_input_nodes[0]
        
        else: 
            raise ValueError
        with g.graph.inserting_before(ap.node): 
            pre_node = g.graph.call_module(pre_target, args=(input_node,))
            ap.node.replace_input_with(input_node , pre_node)
        # Now put quantizer after 

         # add the quantiser to the graph (interposing it between the two linear nodes)
        # We want that after the rewriting each user of `node_pre` reads the
        # output of `new_node` instead; however, in the intermediate state,
        # `new_node` will itself be a user of `node_pre`. Therefore, we need
        # to determine who these users are before `new_node` becomes one of
        # them.
        downstream_nodes = list(ap.node.users)
        with g.graph.inserting_after(ap.node): 
            post_node = g.graph.call_module(post_target , args=(ap.node,)) 
        for u in downstream_nodes:
            u.replace_input_with(ap.node, post_node)
        return g 
                
class DianaF2FInterposer(ComposedEditor): #insert quantidentities between 
    def __init__(self):
        rewriter = Rewriter(name='DianaInterposer', symbolic_trace_fn=quantlib_symbolic_trace,finder= DianaOpQuantFinder(), applier=DianaOpQuantApplier())
        super(DianaF2FInterposer, self).__init__([rewriter ]) 
        
class DianaF2TConverter(ComposedEditor) : 
    def __init__(self) : 
        editors = [
            QuantLibRetracer(),
            F2TAnnotator(),
            EpsTunnelInserter(),
            LinearOpIntegeriser(), 
            DianaRequantizer(),
        
            EpsTunnelConstructSimplifier(),
            EpsTunnelRemover()
        ]

        super(DianaF2TConverter, self).__init__(editors)

    def apply(self,
              g: fx.GraphModule,
              inputdescription: InputDescriptionSpecType = InputDescription(),
              *args,
              **kwargs) -> fx.GraphModule:

        g = self._children_editors[0](g)                    # `QuantLibRetracer`
        g = self._children_editors[1](g, inputdescription)  # `F2TAnnotator`
        for editor in self._children_editors[2:]:
            g = editor(g)

        return g

# Mixed percisionn requantiser 
from quantlib.editing.editing.fake2true.integerisation.requantiser import roles , _BN_KWARGS,_EPS_KWARGS,admissible_screenplays
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern

def get_rewriter_class(class_name: str,
                       pattern: NNSequentialPattern):
    def __init__(self_):
        finder = RequantiserMatcher(pattern)
        applier = DianaRequantizerApplier(pattern )
        NNModuleRewriter.__init__(self_, class_name, pattern, finder, applier)

    class_ = type(class_name, (NNModuleRewriter,), {'__init__': __init__})

    return class_

class DianaLinearOpIntegeriserApplier(Applier ) : 
    pass 
    @staticmethod
    def from_qlinear(qlinear: _QModule) -> nn.Module:
        """Return an ``nn.Module`` implementing a linear operation with
        integerised parameters.
        """
        # TODO: should I offload the responsibility of computing the true-quantised `nn.Module` to `_QLinear`?
        if not isinstance(qlinear, SUPPORTED_LINEAR_FPMODULES):
            raise TypeError
        
        if isinstance(qlinear, nn.Linear):
            class_ = nn.Linear
            new_module = class_(in_features=qlinear.in_features,
                                out_features=qlinear.out_features,
                                bias=qlinear.bias)

        elif isinstance(qlinear, (nn.Conv1d, nn.Conv2d, nn.Conv3d,)):
            if isinstance(qlinear, nn.Conv1d):
                class_ = nn.Conv1d
            elif isinstance(qlinear, nn.Conv2d):
                class_ = nn.Conv2d
            else:  # `isinstance(qlinear, nn.Conv3d)`
                class_ = nn.Conv3d
            new_module = class_(in_channels=qlinear.in_channels,
                                out_channels=qlinear.out_channels,
                                kernel_size=qlinear.kernel_size,
                                stride=qlinear.stride,
                                padding=qlinear.padding,
                                dilation=qlinear.dilation,
                                groups=qlinear.groups,
                                bias=qlinear.bias)

        else:
            raise RuntimeError

        iweight = torch.round(qlinear.qweight.data.clone().detach() / qlinear.scale.data.clone().detach())  # integerised parameters
        # biases are going to be 32 bits in the digital core
        # add qlinear bias later . now it's just the digital core 
        new_module.weight.data = iweight
        new_module.bias.data = qlinear.bias.data.clone().detach() 
        return new_module

class DianaRequantizerApplier(RequantiserApplier): 
    def __init__(self, pattern: NNSequentialPattern):
        super().__init__(pattern, 8) 
    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        # I have to rewrite this 
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_bn         = name_to_match_node['bn'] if 'bn' in name_to_match_node.keys() else None
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_activation = name_to_match_module['activation']
        if node_bn is not None: # what if we have a batch norm followed by a harmonise add. just do it in 8 bits? # assumes bn is always followed by relu 
            self._D = torch.Tensor([2**8]) # 8 bits 
        elif type(module_activation)== DIANAIdentity: 
            self._D = torch.Tensor([module_activation.get_bitwidth()]) 

        #could remove quant layer here if it's taken care of in hardware like in the adc 


        return super()._apply(g, ap, id_)





# create the general-purpose `Requantiser`
class DianaRequantizer(ComposedEditor):
    def __init__(self):
        namespace= {}
        for name, pattern in generate_named_patterns(roles, admissible_screenplays):
            class_name = name + 'Requantiser'

            class_ = get_rewriter_class(class_name, pattern)
            namespace[class_name] = class_   
        super(DianaRequantizer, self).__init__([class_() for class_ in namespace.values()])


    
# if bn is not none then self.d = 8 bits 