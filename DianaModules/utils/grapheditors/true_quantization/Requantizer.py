
#region Requantizer (requantizer at output of operations)

import math
from quantlib.editing.editing.fake2true.integerisation.requantiser import roles ,admissible_screenplays
import torch 
from torch import nn

from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.graphs import fx
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.nametomodule.nametomodule import NameToModule
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from DianaModules.utils.AnalogRequant import AnalogRequantizer
import DianaModules.utils.DigitalRequant as dq

from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher

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

        gamma_int = torch.floor((2**round(math.log2(module_activation.n_levels)) * (eps_in * gamma)             / (sigma * eps_out))) # clip to the power of 2 
        if gamma_int == torch.Tensor([0]) :  # truncation 
            raise RuntimeError('epsilon cannot be quantized with current bitwidth. Something wrong in training phase ')
            #gamma_int = torch.tensor([2**round(math.log2(module_activation.n_levels)) /2])#TODO REMOVE THIS just for testing now
  
        beta_int  = torch.floor((2**round(math.log2(module_activation.n_levels))) * (-mi * gamma + beta * sigma) / (sigma * eps_out))
        div =(2**round(math.log2(module_activation.n_levels)))  / gamma_int
        # create the requantiser
        new_target = id_

        if module_bn is None: 
            new_module = dq.DigitalRequantizer( scale=div, zero=module_activation.zero, n_levels=module_activation.n_levels)
            
        else : 
            new_module = AnalogRequantizer(scale = torch.Tensor([2**round(math.log2(module_activation.n_levels))]) , zero = module_activation.zero , n_levels=module_activation.n_levels, mul=gamma_int , add=beta_int) 
            #raise ValueError
            
        
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