
from DianaModules.utils.Requantizers.muladd import MulAdd
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter

from typing import List
from DianaModules.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.graphs.fx.fxnodes import FXOpcodeClasses
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from torch import nn
import torch 
from DianaModules.core.Operations import AnalogAccumulator
class ResidualAddsAnalogCoreFinder(Finder): # only looks for residual add pattern: 2 inputs , one of which is quantized is passed through eps_tunnel , while the other is a BN. Other cases without the BN are handles with the eps construct simplifier and epstunnel after add
    def __init__(self) -> None:
        super().__init__()
    
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps] = []  
        for n in g .graph.nodes: 
            if ( n.op in FXOpcodeClasses.CALL_FUNCTION.value or n.op in FXOpcodeClasses.CALL_METHOD.value ) and "add" in str(n):
                predecessors = [p for p in n.all_input_nodes]
                if len(predecessors) <= 2: 
                    corrects = 0 
                    try: 
                        if isinstance(g.get_submodule(predecessors[0].target)   , EpsTunnel) : 
                            if isinstance(g.get_submodule(predecessors[1].target)   , nn.BatchNorm2d) : 
                                aps.append(DianaAps('add' , n)) 
                        elif isinstance(g.get_submodule(predecessors[0].target) , nn.BatchNorm2d) :  
                            if isinstance(g.get_submodule(predecessors[1].target) , EpsTunnel) :  
                                aps.append(DianaAps('add' , n)) 
                    except: 
                        continue               
        return aps 
    
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 

class ResidualAddsAnalogCoreApplier(Applier): 
    def __init__(self):
        super().__init__()
        self.bn_bitwidth = torch.Tensor([2**8]) 
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        node = ap.node
        predecessors = [p for p in node.all_input_nodes]
        assert (len(predecessors) <=2) 
        users = [u for u in node.users] 
        node_tunnel =  next((n for n in predecessors if isinstance(g.get_submodule(n.target) ,EpsTunnel)), None)
        node_bn=  next((n for n in predecessors if isinstance(g.get_submodule(n.target) ,nn.BatchNorm2d)), None)
        module_tunnel : EpsTunnel = g.get_submodule(node_tunnel.target)
        module_bn = g.get_submodule(node_bn.target)
        # absorb the scale from the adc 
        nodes_bn_predecessors = [p for p in node_bn.all_input_nodes] 
        nodes_bn_users = [u for u in node_bn.users]  
        assert(len(nodes_bn_users) == 1)
        eps_in = torch.Tensor([1]) 
        assert len(nodes_bn_predecessors) == 1
        try: 
            eps_module = g.get_submodule(nodes_bn_predecessors[0].target)  
            if isinstance(eps_module , AnalogAccumulator): 
        
                acc_predecessors = [p for p in nodes_bn_predecessors[0].all_input_nodes]# predecessors of analog accum 
                assert len(acc_predecessors) ==1 
                e_mod = g .get_submodule(acc_predecessors[0].target) 
                assert isinstance(e_mod, EpsTunnel) 
                eps_module = e_mod 
            if isinstance(eps_module, EpsTunnel):
                eps_in = eps_module._eps_out.clone().detach() 
                eps_module.set_eps_out(torch.ones_like(eps_module._eps_out)) 
            
        except: 
            pass 
        #compute mul add with the scale 
        assert(node_tunnel is not None and node_bn is not None)
        
        shape = node_bn.meta['tensor_meta'].shape
        mi      = module_bn.running_mean 
        sigma   = torch.sqrt(module_bn.running_var + module_bn.eps) 
        gamma   = module_bn.weight 
        beta    = module_bn.bias
        broadcast_shape = tuple(1 if i != 1 else mi.numel() for i, _ in enumerate(range(0, len(shape))))
        mi    = mi.reshape(broadcast_shape)
        sigma = sigma.reshape(broadcast_shape)
        gamma = gamma.reshape(broadcast_shape)
        beta  = beta.reshape(broadcast_shape)

        # Mul gamma / sigma 
        # Add beta    = module_bn.bias
        mul = eps_in *gamma /sigma 
        add =(-mi * gamma + beta * sigma) / (sigma ) 
        factored_power_of_2 = torch.Tensor([1])
        while True  : 
            max_multiplier = torch.max(torch.maximum(torch.abs(mul ), torch.abs(add )))
            factored_power_of_2 += 1
            if  torch.exp2(factored_power_of_2) * max_multiplier > self.bn_bitwidth/2 :

                factored_power_of_2 -=1
                break 
            
        add =torch.floor( torch.exp2(factored_power_of_2) *  add )
        mul = torch.floor(torch.exp2(factored_power_of_2) * mul ) 
        # add new Mul add module and delete batch norm 
        mul_add_target = id_ 
        muladd_module = MulAdd(mul , add ) 
        g.add_submodule(mul_add_target, muladd_module) 
        with g.graph.inserting_after(node_bn ): 
            mul_add_node = g.graph.call_module(mul_add_target , (nodes_bn_predecessors[0], )) 
        node.replace_input_with(node_bn , mul_add_node)      
        g.delete_submodule(node_bn.target) 
        g.graph.erase_node(node_bn) 

        # Shift other input of addition by scale and insert eps tunnel after the addition 
        scale_factor = torch.exp2(factored_power_of_2)
        module_tunnel.set_eps_out(torch.clamp(torch.exp2(torch.round(torch.log2(module_tunnel._eps_out *scale_factor) )), min=torch.Tensor([1])))
        #print("Module_tunnel scale: " , module_tunnel._eps_out) 
        
        eps_out_target = id_ +  f'[{str(self._counter)}]'
        new_module = EpsTunnel(torch.Tensor([1]))
        new_module.set_eps_in(scale_factor)
        g.add_submodule(eps_out_target , new_module)
        with g.graph.inserting_after(node): 
            out_eps_node = g.graph.call_module(eps_out_target, (node, )) 
        for u in users: 
            u.replace_input_with(node , out_eps_node)
        return g 

class ResidualAddsAnalogCoreRewriter(Rewriter) : 
    def __init__(self):
        super(ResidualAddsAnalogCoreRewriter, self).__init__(name='ResidualAnalogAdditions', symbolic_trace_fn=quantlib_symbolic_trace,finder= ResidualAddsAnalogCoreFinder(), applier=ResidualAddsAnalogCoreApplier())
