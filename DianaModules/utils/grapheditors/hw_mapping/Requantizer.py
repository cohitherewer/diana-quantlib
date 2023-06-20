from random import randint
import torch 
from torch import nn
from DianaModules.core.Operations import AnalogAccumulator, AnalogOutIdentity
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule

from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier


from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
import torch.fx as fx
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap

from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.nnmodules import NNSequentialPattern
from DianaModules.utils.Requantizers.AnalogRequant import AnalogRequantizer
import DianaModules.utils.Requantizers.DigitalRequant as dq

from quantlib.editing.editing.fake2true.integerisation.requantiser.finder import RequantiserMatcher
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
_BN_KWARGS = {'num_features': 1}
_EPS_KWARGS = {'eps': torch.Tensor([1.0])}
checker = (lambda m : True if type(m) != AnalogOutIdentity else False, )

roles = Roles([

    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
        ('',     None),  # This line must be last to give priority to matching with Eps first
    ])),
    ('accumulator',  Candidates([
        ('',     None),  #analog accumulator is an optional part of the pattern
        ('AniaAccumulator', NNModuleDescription(class_=AnalogAccumulator, kwargs={})),
    ])),
    ('bn', Candidates([
        ('',     None),  # batch-normalisation is an optional part of the pattern
        ('BN1d', NNModuleDescription(class_=nn.BatchNorm1d, kwargs=_BN_KWARGS)),
        ('BN2d', NNModuleDescription(class_=nn.BatchNorm2d, kwargs=_BN_KWARGS)),
        ('BN3d', NNModuleDescription(class_=nn.BatchNorm3d, kwargs=_BN_KWARGS)),
    ])),

    ('activation', Candidates([
        ('QIdentity',  NNModuleDescription(class_=nn.Identity,  kwargs={} , checkers=checker)),
        ('QReLU',      NNModuleDescription(class_=nn.ReLU,      kwargs={})),
        ('QReLU6',     NNModuleDescription(class_=nn.ReLU6,     kwargs={})),
        ('QLeakyReLU', NNModuleDescription(class_=nn.LeakyReLU, kwargs={})),
    ])),

    ('eps_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs=_EPS_KWARGS)),
    ])),
])
admissible_screenplays = list(roles.all_screenplays)
class DianaRequantizerApplier(NNModuleApplier): # this will probably have to be rewritten 
    def __init__(self,
                 pattern: NNSequentialPattern,
                ):  # the integer bit-shift parameter

        super(DianaRequantizerApplier, self).__init__(pattern)
        self.div_max_bitwidth = torch.Tensor([2 ** 15])  # the requantisation factor
        self.bn_bitwidth =torch.Tensor([2**8]) 
        self.counter = 0 
    

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        
        node_bn         = name_to_match_node['bn'] if 'bn' in name_to_match_node.keys() else None
        node_acc         = name_to_match_node['accumulator'] if 'accumulator' in name_to_match_node.keys() else None
        node_activation = name_to_match_node['activation']

        node_eps_in     = name_to_match_node['eps_in'] if 'eps_in' in name_to_match_node.keys() else [p for p in node_activation.all_input_nodes][0] #if eps in isint there then accumulator isint there so it's safe to use the input to activation
        
        node_eps_out    = name_to_match_node['eps_out']
        
        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in     = name_to_match_module['eps_in'] if 'eps_in' in name_to_match_module.keys() else None
        module_bn         = name_to_match_module['bn'] if 'bn' in name_to_match_module.keys() else None
        module_activation = name_to_match_module['activation']
      
        module_eps_out    = name_to_match_module['eps_out']
        if (module_activation is None): 
            return g 
        assert ((node_bn is None) and (module_bn is None)) or (isinstance(node_bn, fx.Node) and isinstance(module_bn, nn.Module))

        # extract the parameters required to compute the requantiser's parameters
        eps_in  = module_eps_in.eps_out if module_eps_in else torch.Tensor([1])
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
        new_target = id_ #+ f"[{self.counter}]"
        #self.counter += 1 
        if module_bn is None: 
            
            gamma_int = torch.floor( self.div_max_bitwidth *eps_in             / ( eps_out))# mul then div by self.D b
            

            if  torch.all(gamma_int.eq(torch.Tensor([0]))) :  # truncation 
                raise RuntimeError('epsilon cannot be quantized with current bitwidth. Something wrong in training phase ')
            #if '[9]' not in node_activation.target: 
            div = eps_out / eps_in
                
            #else: 
            #    pred = [p for p in node_activation.all_input_nodes] [0]
            #    pred_mode = g.get_submodule(pred.target)
            #    users = [ u for u in node_activation.users] [0]
            #    user_mode = g.get_submodule(users.target)
            #    print(f"Node {node_activation} has eps_tunnel pred with eps_out {pred_mode.eps_out} and user with eps_in {user_mode.eps_in}")
                # without relu 1  is 0.8108
                # without relu 2  is 0.7975 (original acc)
                # without relu 3  is 0.7999
                # without relu 4 is 0.7801
                # without relu mod 9 biggest difference 8508 

          #  print("without rounding to 2" , self.div_max_bitwidth  / gamma_int, " eps_in : " , eps_in  , " eps_out : ", eps_out) # TODO relus scales weren't pow2
            #print("Maximum difference between without rounding and rounding: " ,  torch.max(div - self.div_max_bitwidth  / gamma_int) , f" for activation node: {node_activation}")

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
            gamma_int = (eps_in * gamma)             / (sigma) # clip to the power of 2 . eps_in is the ADC scale 

            beta_int  =( -mi * gamma + beta * sigma) / (sigma )
            #print("max gamma_int" ,torch.max(torch.abs(gamma_int )))
            #print("max beta int " ,torch.max(torch.abs(beta_int )))
            #print("Eps out: " ,torch.log2(eps_out))
            #print("eps in ; " , torch.log2(eps_in))
            factored_power_of_2 = torch.Tensor([15])

            while True  : 
                max_multiplier = torch.max(torch.maximum(torch.abs(gamma_int ), torch.abs(beta_int )))
                
                if  torch.exp2(factored_power_of_2) * max_multiplier >= self.bn_bitwidth /2:
                     
                    factored_power_of_2 -=1
                else: 
                   # print( "Maximum value: " , max_multiplier * torch.exp2(factored_power_of_2))
                   # print("factored power of 2" , factored_power_of_2)
                    break 
            
            beta_int =torch.clamp(torch.round( torch.exp2(factored_power_of_2) *  beta_int ) , min = -self.bn_bitwidth /2, max = self.bn_bitwidth/2 - 1)
            gamma_int = torch.clamp(torch.round(torch.exp2(factored_power_of_2) * gamma_int ) , min = -self.bn_bitwidth /2, max = self.bn_bitwidth/2 - 1)
            div = torch.clamp(torch.exp2(factored_power_of_2) * eps_out , min=torch.Tensor([1]))
            #print("after factoring gamma_int is (mul/div): " , gamma_int/div)
            #print("after factoring beta_int is (mul/div): " , beta_int/div)

            new_module = AnalogRequantizer(div,  module_activation.zero , module_activation.n_levels, gamma_int , beta_int) 

        # add the requantiser to the graph...
        g.add_submodule(new_target, new_module)
        if node_acc is not None: 
            with g.graph.inserting_after(node_acc):
                new_node = g.graph.call_module(new_target, args=(node_acc,))
        else: 
            with g.graph.inserting_after(node_eps_in):
                new_node = g.graph.call_module(new_target, args=(node_eps_in,))
            
        node_eps_out.replace_input_with(node_activation, new_node)
        if module_eps_in: 
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
