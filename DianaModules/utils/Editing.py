
import itertools
import math
from tabnanny import check
from typing import List, OrderedDict, Union

import torch

from DianaModules.core.Operations import DIANAIdentity, DIANALinear, DIANAReLU, IdentityType , DIANAConv2d
import DianaModules.utils.DigitalRequant as dq
from quantlib.algorithms.qmodules.qmodules.qmodules import _QActivation, _QModule

from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor

from quantlib.editing.editing.editors.base.rewriter.applicationpoint import ApplicationPoint
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher
from quantlib.editing.editing.editors.nnmodules.pattern.base.pattern import NNModulePattern
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from DianaModules.utils.AnalogRequant import AnalogRequantizer
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.editing.editors.nnmodules.rewriter.rewriter import NNModuleRewriter
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.fake2true.annotation import F2TAnnotator
from quantlib.editing.editing.fake2true.annotation.inputdescription import InputDescription, InputDescriptionSpecType
from quantlib.editing.editing.fake2true.epstunnels.inserter.rewriter import EpsTunnelInserter
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import EpsTunnelRemover
from quantlib.editing.editing.fake2true.epstunnels.simplifier.rewriter import EpsTunnelConstructSimplifier
from quantlib.editing.editing.fake2true.integerisation.linearopintegeriser.finder import LinearOpIntegeriserMatcher

from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher
from quantlib.editing.editing.float2fake.canonicalisation import F2FCanonicaliser

from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.retracer import QuantLibHarmonisedAddRetracer
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.rewriter import AddTreeHarmoniser

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescriptionSpecType
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.nametomodule.nametomodule import NameToModule
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.rewriter import ModuleWiseConverter
from quantlib.editing.editing.float2fake.quantisation.qdescription.qdescription import QDescriptionSpecType

import torch.fx as fx
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
# Mixed percisionn requantiser 

from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel

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
       
           
            
        
            DianaF2FInterposer()  , 
            
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            ),
            QuantLibRetracer()  ,
            DianaQuantizerFuser() ,# ignore the harmonise adds 
           
           
        ]) # Add interposer here 

# each conv layer will have a a value indicating if the output is used in activation (ReLU) . if there is then we put a a ReLU activation afterwards. otherwise it's just the identity ( This is done in the interposer )


#region FakeBatchNormsInsertion 

#endregion 


#region DianaActivationFuser
#### Fuser ### 
# there will always be an output quantizer for analog conv. #fuse div clips than onlu have 1 out and their bitwidth is the same (scale of 1 of 2nd activations)

class DianaQuantizerFuserFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps]
        aps = [] 
        #name_to_module #get names and modules with activations 
        # get modules first 
        names = []
        for name ,_ in g.named_modules(): 
            if(issubclass(type(g.get_submodule(name)) , _QActivation)): 
                names.append(name)
            
        for node in g.graph.nodes: 
            if node.target in names : 
                predecessor = [ p for p in node.all_input_nodes ] 
                if len(predecessor) == 1 and predecessor[0].target in names:# and  issubclass(type(g.get_submodule(users[0].target)), _QActivation): 
                    aps.append(DianaAps('' , node))
        
        return aps 
    def check_aps_commutativity(self, aps: List[ApplicationPoint]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 
class DianaQuantizerFuserApplier(Applier) : 
    def __init__(self):
        super().__init__()
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        pre_node = [p for p in ap.node.all_input_nodes][0]
        pre_module = g.get_submodule(pre_node.target) 
        cur_module = g.get_submodule(ap.node.target) 
        lower_bitwidth = 0 if (min(pre_module.n_levels, cur_module.n_levels) == pre_module.n_levels ) else 1
        if lower_bitwidth == 0 : 
            #if it's 0 then remove cur node 
            cur_node_users = [u for u in ap.node.users ]

            for p_node in cur_node_users: 
                p_node.replace_input_with(ap.node, pre_node)
            pre_module.clip_lo = max(pre_module.clip_lo , cur_module.clip_lo)
            pre_module.clip_hi = min(pre_module.clip_hi , cur_module.clip_hi)
            g.delete_submodule(ap.node.target)
            g.graph.erase_node(ap.node)
            # Edit clipping of pre_node        
        else: 
            pre_node_users = [p for p in pre_node.users]  # upstream
            if len(pre_node_users)  == 1:  # make sure pre node only has 1 user
                for user in pre_node_users:  #without ap.node
                    user.replace_input_with(pre_node, ap.node)
                cur_module.clip_lo = max(pre_module.clip_lo , cur_module.clip_lo)
                cur_module.clip_hi = min(pre_module.clip_hi , cur_module.clip_hi)
                g.delete_submodule(pre_node.target)
                g.graph.erase_node(pre_node)
        
        return g 
#this class is used for fusing activations in true to fake 
class DianaQuantizerFuser(Rewriter) :
    def __init__(self):
        super().__init__("DianaActivationFuser", quantlib_symbolic_trace, DianaQuantizerFuserFinder(), DianaQuantizerFuserApplier())
     
#endregion

# Modules to look for DIANAConv2d , DIANALinear , AvgPool2d  
from torch import  Tensor, nn 

#region Diana fake to true neural nets 

class DianaF2TConverter(ComposedEditor) : 
    def __init__(self, custom_editor : List[Editor] = []) : 
        editors = [
            
            QuantLibRetracer(),
          
            F2TAnnotator(),
            EpsTunnelInserter(),
         
            DianaLinearOpIntegrizer(), 
         
            DianaRequantizer()]
        
        editor_post = [   
            EpsTunnelConstructSimplifier(),
            EpsTunnelRemover()
        ]

        super(DianaF2TConverter, self).__init__(editors + custom_editor + editor_post)

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


#region DianaquantizersInterposer

MODULES_WITH_QUANTIZERS = [DIANAConv2d , DIANALinear]# ,nn.AvgPool2d , nn.AdaptiveAvgPool2d]

class DianaOpQuantFinder(Finder):

    def __init__(self):
        super().__init__()
    
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        def getmodule_name(module) :
            if type(module) == DIANAConv2d: 
                return 'conv' 
            elif type(module) == DIANALinear: 
                return 'linear'
            
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
            type_out = IdentityType.AIMC_OUT #uncomment later when using analog core 
        qpre = DIANAIdentity({'bitwidth': 8, 'signed': True} , 'per-array', 'minmax', type_in)
        qpost = DIANAIdentity({'bitwidth':8 , 'signed': True} , 'per-array', 'minmax', type_out) if type_out == IdentityType.AIMC_OUT else None
 
        pre_target = id_ 
        
        
        

        g.add_submodule(pre_target ,qpre) 
        
            
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
        if qpost is not None: 
            post_target = id_ + f'[{str(self._counter)}]'
            g.add_submodule(post_target, qpost) 
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

#endregion 




#region LinearOpIntegrizer
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
di_roles = Roles([

    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),

    ('linear', Candidates([
        ('QLinear', NNModuleDescription(class_=nn.Linear, kwargs={'in_features': 1, 'out_features': 1, 'bias': True}, checkers=checker)) , 
        ('QConv2d', NNModuleDescription(class_=nn.Conv2d , kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'bias': True}, checkers=checker))
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
   



########################### Mixed Percision Requant ###########################

#region Requantizer (requantizer at output of operations)

from quantlib.editing.editing.fake2true.integerisation.requantiser import roles , _EPS_KWARGS,admissible_screenplays

class DianaRequantizerApplier(NNModuleApplier): # this will probably have to be rewritten 
    def __init__(self, pattern: NNSequentialPattern):
        super().__init__(pattern) 
    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in     = name_to_match_node['eps_in']
        node_bn         = name_to_match_node['bn'] if 'bn' in name_to_match_node.keys() else None
        node_activation = name_to_match_node['activation']
        node_eps_out    = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in     = name_to_match_module['eps_in']
        module_bn         = name_to_match_module['bn'] if 'bn' in name_to_match_module.keys() else None
        module_activation = name_to_match_module['activation']
        module_eps_out    = name_to_match_module['eps_out']
        if(issubclass(type(module_activation), dq.DigitalRequantizer) or isinstance(module_activation, AnalogRequantizer)): 
            # return same module from nn.sequential 
            return g
        assert ((node_bn is None) and (module_bn is None)) or (isinstance(node_bn, fx.Node) and isinstance(module_bn, nn.Module))

        # extract the parameters required to compute the requantiser's parameters
        eps_in  = module_eps_in.eps_out
        mi      = module_bn.running_mean if module_bn is not None else torch.zeros_like(eps_in)
        sigma   = torch.sqrt(module_bn.running_var + module_bn.eps) if module_bn is not None else torch.ones_like(eps_in)
        gamma   = module_bn.weight if module_bn is not None else torch.ones_like(eps_in)
        beta    = module_bn.bias if module_bn is not None else torch.zeros_like(eps_in)
        eps_out = module_eps_out.eps_in
        assert torch.all(eps_out == module_activation.scale)

        # compute the requantiser's parameters
        shape = node_activation.meta['tensor_meta'].shape
        broadcast_shape = tuple(1 if i != 1 else mi.numel() for i, _ in enumerate(range(0, len(shape))))
        mi    = mi.reshape(broadcast_shape)
        sigma = sigma.reshape(broadcast_shape)
        gamma = gamma.reshape(broadcast_shape)
        beta  = beta.reshape(broadcast_shape)

        gamma_int = torch.floor((2**round(math.log2(module_activation.n_levels))) * (eps_in * gamma)             / (sigma * eps_out)) # clip to the power of 2 
        if gamma_int == torch.Tensor([0]) :  # truncation 
            raise RuntimeError('epsilon cannot be quantized with current bitwidth. Something wrong in training phase ')
  
        beta_int  = torch.floor((2**round(math.log2(module_activation.n_levels))) * (-mi * gamma + beta * sigma) / (sigma * eps_out))
        div =(2**round(math.log2(module_activation.n_levels)))  / gamma_int
        # create the requantiser
        new_target = id_

        if module_bn is None: 
            new_module = dq.DigitalRequantizer( scale=div, zero=module_activation.zero, n_levels=module_activation.n_levels)
            
        else : 
            #new_module = dq.DigitalRequantizer( scale=div, zero=module_activation.zero, n_levels=module_activation.n_levels)# as a test remove this later 
            #new_module = AnalogRequantizer() 
            raise ValueError
            
        
        # add the requantiser to the graph...
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_activation, new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))

        # ...and delete the old construct
        g.delete_submodule(node_activation.target)
        g.graph.erase_node(node_activation)  # since `node_activation` is a user of `node_bn`, we must delete it first
        
        name_to_module = NameToModule(g.named_modules())
        name_to_module[node_activation.target] = new_module
        path_to_parent, child = NameToModule.split_path_to_target(node_activation.target)
        setattr(name_to_module[path_to_parent], child, new_module)  # this is needed for nn sequentials

        if node_bn is not None:
            g.delete_submodule(node_bn.target)
            g.graph.erase_node(node_bn)
            #path_to_parent, child = NameToModule.split_path_to_target(node_bn.target)
            #delattr(name_to_module[path_to_parent], child)
        return g

# create the general-purpose `Requantiser`
class DianaRequantizer(ComposedEditor):
    def __init__(self):
        namespace= {}
        for name, pattern in generate_named_patterns(roles, admissible_screenplays):
            class_name = name + 'Requantiser'

            class_ = get_rewriter_class(class_name, pattern, RequantiserMatcher, DianaRequantizerApplier)
            namespace[class_name] = class_   
        super(DianaRequantizer, self).__init__([class_() for class_ in namespace.values()])

#endregion 

#region RequantizerSimplifier
class DianaRequantizerSimplifierFinder(Finder): 
    pass 
class DianaRequantizerSimplifierApplier(Applier): 
    pass 
class DianaRequantizerSimplifier(Rewriter): 

    pass 

#endregion


