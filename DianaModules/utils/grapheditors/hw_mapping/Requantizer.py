
#region Requantizer (requantizer at output of operations)

import math
from quantlib.editing.editing.fake2true.integerisation.requantiser import roles ,admissible_screenplays
import torch 
from torch import nn

from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
import torch.fx as fx
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.nametomodule.nametomodule import NameToModule
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from DianaModules.utils.AnalogRequant import AnalogRequantizer
import DianaModules.utils.DigitalRequant as dq

from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher
from quantlib.editing.graphs.nn.requant import Requantisation

class DianaRequantizerApplier(NNModuleApplier): # this will probably have to be rewritten 
    def __init__(self,
                 pattern: NNSequentialPattern,
                ):  # the integer bit-shift parameter

        super(DianaRequantizerApplier, self).__init__(pattern)
        self.div_max_bitwidth = torch.Tensor([2 ** 15])  # the requantisation factor
        self.bn_bitwidth =torch.Tensor([2**8]) 
    

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

        #gamma_int = torch.floor(self.D * (eps_in * gamma)             / (sigma * eps_out))
        #beta_int  = torch.floor(self.D * (-mi * gamma + beta * sigma) / (sigma * eps_out))

        # create the requantiser
        new_target = id_
        if module_bn is None: 
            
          
            gamma_int = torch.floor( self.div_max_bitwidth *eps_in             / ( eps_out))# mul then div by self.D b
            

            if gamma_int == torch.Tensor([0]) :  # truncation 
                raise RuntimeError('epsilon cannot be quantized with current bitwidth. Something wrong in training phase ')
            div = torch.exp2(torch.round(torch.log2(self.div_max_bitwidth  / gamma_int)))
            #print("without rounding to 2" , self.div_max_bitwidth  / gamma_int, " eps_in : " , eps_in  , " eps_out : ", eps_out) # TODO relus scales weren't pow2
            #print("with rounding to 2" ,  torch.exp2(torch.round(torch.log2(self.div_max_bitwidth  / gamma_int))))
            new_module = dq.DigitalRequantizer( div=div, zero=module_activation.zero, n_levels=module_activation.n_levels)
        else: 
            # for onnx graph generation later 
            #gamma_int  = torch.floor(self.bn_bitwidth * gamma / sigma)   / self.bn_bitwidth
            # end 
            #print("sigma is" , sigma)
            #print("eps_out is " , eps_out)
            #print("sigma * eps_out ", sigma)
            
            gamma_int = torch.floor(self.bn_bitwidth * (eps_in * gamma)             / (sigma * eps_out)) #clip to power of 2
           
            if torch.all(gamma_int.eq(torch.Tensor([0])) ):  # truncation 
                raise RuntimeError('epsilon cannot be quantized with current bitwidth. Something wrong in training phase ')
            beta_int  = torch.floor(self.bn_bitwidth * (-mi * gamma + beta * sigma) / (sigma * eps_out))
            #print("Before factoring gamma_int is (mul/div): " , gamma_int/self.bn_bitwidth)
            #print("Before factoring beta_int is (mul/div): " , beta_int/self.bn_bitwidth)
            gamma_int = (eps_in * gamma)             / (sigma * eps_out) #clip to power of 2
            beta_int  =( -mi * gamma + beta * sigma) / (sigma * eps_out)
            factored_power_of_2 = torch.Tensor([1])
            while True  : 
                max_multiplier = torch.max(torch.maximum(torch.abs(gamma_int ), torch.abs(beta_int )))
                factored_power_of_2 += 1
                if  torch.exp2(factored_power_of_2) * max_multiplier > self.bn_bitwidth/2 :
                     
                    factored_power_of_2 -=1
                    break 
            
            beta_int =torch.floor( torch.exp2(factored_power_of_2) *  beta_int )
            gamma_int = torch.floor(torch.exp2(factored_power_of_2) * gamma_int ) 
            div = torch.exp2(factored_power_of_2)
            #print("after factoring gamma_int is (mul/div): " , gamma_int/div)
            #print("after factoring beta_int is (mul/div): " , beta_int/div)

            new_module = AnalogRequantizer(div,  module_activation.zero , module_activation.n_levels, gamma_int , beta_int) 

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
        if node_bn is not None:
            g.delete_submodule(node_bn.target)
            g.graph.erase_node(node_bn)
        
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
