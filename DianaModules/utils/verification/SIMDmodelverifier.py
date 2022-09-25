from importlib.metadata import EntryPoint
import torch.fx as fx 
from typing import Dict, Optional, Union  , List
from dataclasses import dataclass
import numpy as np 
import torch 
from quantlib.editing.editing.editors.base.rewriter import Finder
from DianaModules.utils.grapheditors import DianaAps
from DianaModules.core.Operations import AnalogAccumulator, AnalogConv2d, AnalogGaussianNoise, AnalogOutIdentity
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from ..Requantizers.DigitalRequant import DigitalRequantizer
from ..Requantizers.AnalogRequant import AnalogRequantizer 
from ..Requantizers.muladd import MulAdd 
from quantlib.editing.graphs.fx import FXOpcodeClasses
from ...HWModel.HWModel import SIMDModel
SIMD_VERIFIER_TAG = "SIMD_VERIFIER: "
#TODO Absorb eps tunnel into requantizer 
VALID_PATTERNS = [[AnalogAccumulator, DigitalRequantizer] , [AnalogAccumulator, AnalogRequantizer] , [AnalogAccumulator, MulAdd , 'add' , EpsTunnel, DigitalRequantizer], [AnalogAccumulator, 'add' ,EpsTunnel, DigitalRequantizer] ] 
#pattern = [Analog Core Conv2d(noise , ADC), BN (optional)  ,res_add(optional) ,  Activation ] , #ifnot epstunnels , noise , adc , accumulator 
#Verification happens fter model is HW Mapped , Look for AnalogConv2d -> DigitalRequantizer , AnalogRequantizer , MUL Add resadd activation , #Skip epstunnels , noise node , and adc 
class SIMDnodesFinder(Finder) : 
    def __init__(self) -> None:
        super().__init__()
    def find(self, g: fx.GraphModule) -> List[List[DianaAps]]:
        aps : List[List[DianaAps]] = [] 
        for node in g.graph.nodes: 
            try: 
                if isinstance(g.get_submodule(node.target) , AnalogAccumulator) : 
                    # traverse forward 
                    current_node = node
                    pattern = [DianaAps('Accumulator' , current_node)] 
                    for i in range(4) : # max depth 4 
                        user : fx.node.Node = [ u for u in current_node.users]  [0]
                        current_node = user 
                        if user.op in FXOpcodeClasses.CALL_METHOD.value and 'add' in user.target: 
                            pattern.append(DianaAps('add' , user))
                            continue
                        elif user.op in FXOpcodeClasses.CALL_FUNCTION.value and 'add' in user.target.__name__: 
                            pattern.append(DianaAps('add' , user))
                            continue
                        module = g.get_submodule(user.target ) 
                        
                        if isinstance(module , (DigitalRequantizer , AnalogRequantizer)): 
                            pattern.append(DianaAps('Requantizer' ,user)) 
                            break 
                        else: #  Mul Add  , Res Add , Activation 
                            pattern.append(DianaAps(str(type(module).__name__) ,user)) 
                    assert len(pattern) <= 5# max lenght is Accumulator , muladd modul , add , requantizer so 3 
                    aps.append(pattern) 

            except :
                #print(SIMD_VERIFIER_TAG, f"Tried getting submodule of node {node} but it wasn't found.")
                pass  

        return aps
    def check_aps_commutativity(self, aps: List[List[DianaAps]]) -> bool:
        return len(aps) == len(set(ap.node for lap in aps for ap in lap))  # each `fx.Node` should appear at most once 


@dataclass
class SIMDLayer: 
    bn_w : Union[np.ndarray , None] =None#quantized batchnorm weights
    bn_ws : Union[np.ndarray , int] =0 #batchnorm weight scale 
    bn_b : Union[np.ndarray , None] =None #batchnorm bias 
    bn_bs : Union[np.ndarray , int] =0 #batchnorm bias scale 2**b
    r_s : int =1  # scale of tensor 2**r_s
    q_s : int =0 # quantization scale  2**q_s
    relu : bool = False 
    bitwidth_clip: int = 8 
    res_in: Optional[np.ndarray] = None
    def print_params(self): 
        print(f'bn_w: {self.bn_w} \nbn_ws: {self.bn_ws} \nbn_b: {self.bn_b} \nbn_bs: {self.bn_bs} \nr_s: {self.r_s} \nq_s: {self.q_s} ')
        
class SIMDVerifier: 
    def __init__(self, model : fx.graph_module.GraphModule) -> None:
        self.model = model 
        self.model_input_output ={}
        self.SIMDparameters : List[SIMDLayer]= []
        self.aps : List[List[DianaAps]]= [] 
        assert isinstance(model,fx.graph_module.GraphModule)
    def verify(self, input : torch.Tensor ) -> None: 
        # Get application points 
        finder = SIMDnodesFinder()
        self.aps =finder.find(self.model) 
        # validate patterns 
        for pattern in self.aps: 
            assert self.isvalid_pattern(pattern) 
        # register hooks and FP 
        self.register_hooks() 
        out = self.model(input)
        # extract parameters and inputs 
        for pattern in self.aps: 
            self.SIMDparameters.append(self.extract_features_from_pattern(pattern)) 
        # iteratively compare HWmodel output and model's output 
        for idx, node in enumerate(self.model_input_output.keys()):  
            resadd = self.SIMDparameters[idx].res_in 
            SIMD_out = SIMDModel.simd_op(self.SIMDparameters[idx], self.model_input_output[node]['input'].numpy() , resadd)
            model_out =self.model_input_output[node]['output'].numpy()
            max_diff = np.max(abs(SIMD_out - model_out))
            assert np.allclose(model_out,SIMD_out )
        return True  
    def isvalid_pattern(self, pattern : List[DianaAps]) -> bool: # this is redundant but good for 
        is_valid = True
        for valid_pattern in VALID_PATTERNS:  
            is_valid =True
            if len(pattern) != len(valid_pattern): 
                continue 
            index = 0 
            for ap in pattern: 
                node = ap.node 
                if node.op in FXOpcodeClasses.CALL_MODULE.value:
                    module = self.model.get_submodule(node.target) 
                    if not isinstance(module , valid_pattern[index]): 
                        is_valid= False 

                elif node.op in FXOpcodeClasses.CALL_METHOD.value:
                    if not valid_pattern[index] in node.target : 
                        is_valid= False 
                elif node.op in FXOpcodeClasses.CALL_FUNCTION.value:
                    if not valid_pattern[index] in node.target.__name__ :  
                        is_valid= False 
                index +=1 
        return is_valid 
    def extract_features_from_pattern (self, pattern : List[DianaAps]) -> SIMDLayer: 
        layer = SIMDLayer() 
        for ap in pattern:
            if ap.type == 'Requantizer' : 
                # extract quantization , relu and BN if analog BN 
                module = self.model.get_submodule(ap.node.target) 
                layer.bitwidth_clip = torch.round(torch.log2(module.clip_hi - module.clip_lo)).item()
                layer.q_s += torch.log2(module.div ).item() 
                if module.clip_lo == torch.Tensor([0]) : 
                    layer.relu = True 
                    layer.bitwidth_clip +=1 # this is because in the HW models the bw factor is subtracted by 1 
                    
                if isinstance(module, AnalogRequantizer): # extract bn 
                    layer.bn_w =module.mul .numpy()
                    layer.bn_b = module.add .numpy() 
            elif ap.type == 'MulAdd': 
                #extract bn_w  , bn_ws , bn_b  ,bn_bs 
                module = self.model.get_submodule(ap.node.target) 
                layer.bn_w = module.mul.numpy()
                layer.bn_b = module.add .numpy()
            elif ap.node.op in FXOpcodeClasses.CALL_METHOD.value or ap.node.op in FXOpcodeClasses.CALL_FUNCTION.value :
                # extract res_in r_s (EpsTunnel)
                res_in_module : EpsTunnel= self.model.get_submodule(next(iter(filter(lambda node: isinstance(self.model.get_submodule(node.target) ,EpsTunnel)  , ap.node.all_input_nodes)))  .target)
                layer.r_s = torch.log2(self.model_input_output[pattern[0].node.target]['resadd_scale']).item()
                layer.res_in = (self.model_input_output[pattern[0].node.target]['resadd']/self.model_input_output[pattern[0].node.target]['resadd_scale']).numpy() 
            elif ap.type == 'EpsTunnel': 
                module = self.model.get_submodule(ap.node.target) 
                layer.q_s += torch.round(torch.log2(module.eps_out/module.eps_in)).item() 
                

                 
              



        return layer
    def verify_SIMDop(self) : 
        pass 
    def register_hooks(self)  : 
        for l in self.aps: 
            accumulator_module = self.model.get_submodule(l[0].node.target) 
            self.model_input_output[l[0].node.target] = {} 
            assert type(accumulator_module) == AnalogAccumulator
            accumulator_module.register_forward_hook(self.register_out_hook(l[0].node.target, 'input') ) 
            activation_module = self.model.get_submodule(l[len(l)-1].node.target)  
            activation_module.register_forward_hook(self.register_out_hook(l[0].node.target, 'output'))
            # res add 
            for i in range(len(l )):  
                ap = l[i]
                node = ap.node 
                if node.op in FXOpcodeClasses.CALL_METHOD.value or node.op in FXOpcodeClasses.CALL_FUNCTION.value : #addition 
                    input_nodes = [p for p in node.all_input_nodes] 
                    for input in input_nodes: 
                        if isinstance(self.model.get_submodule(input.target) , EpsTunnel) : 
                            resadd_module = self.model.get_submodule(input.target)
                            break 
                    resadd_module.register_forward_hook(self.register_out_hook(l[0].node.target, 'resadd'))
                    self.model_input_output[l[0].node.target]['resadd_scale'] = resadd_module.eps_out / resadd_module.eps_in
                    break 






        
    def register_out_hook(self ,key , in_or_out :str = 'input'): 
        def hook(model, input, output):
            self.model_input_output[key][in_or_out] = output.detach()
        return hook

    pass 
