import torch
from torch import nn 
from typing import Any, Tuple, Union
import quantlib.editing.graphs as qg
from quantlib.editing.graphs.nn.requant import Requantisation 
class AnalogFeatureExtractor: # Linear layers followed by identities
    def __init__(self , model : nn.Module ) -> None:
       
        self.model = model 
        self.graph_mod = qg.fx.quantlib_symbolic_trace(root=model) # graph module 
        self.model_input_output = []
    def register_hooks(self): 
        pass
    def extract(self, x : torch.Tensor): 
        for node in self.graph_mod.graph.nodes: 
            try: 
                module = self.graph_mod.get_submodule(node.target)
                if (module.is_analog == torch.Tensor([True]) ):
                    
                    user = node
                    pred = [p for p in node.all_input_nodes][0]
                    pred_mod=self.graph_mod.get_submodule(pred.target)
                    while True: 
                        users =[u for u in user.users] 
                        user= users[0] 
                        user_mod = self.graph_mod.get_submodule(user.target)
                        if(isinstance(user_mod, Requantisation)): 
                            break 
                    #print(user_mod)
                    self.model_input_output.append({node.target:{"scale": user_mod.div,"weight": module.weight}}) 
                    pred_mod.register_forward_hook(self._out_hook(node.target, len(self.model_input_output)-1 , "input"))
                    user_mod.register_forward_hook(self._out_hook(node.target, len(self.model_input_output)-1 , "output"))
                    
            except: 
                pass 
        
        with torch.no_grad(): 
            _ = self.model(x)
         
    def _out_hook(self ,key, index , in_out ): 
        def hook(model, input, output):
            self.model_input_output[index][key][in_out] = output.detach()
        return hook
    def serialize(self , filepath): 
        assert(self.model_input_output != {})
        with open(str(filepath), 'w') as file:
            for e in self.model_input_output: 
                layer_name = list(e.keys())[0] #only 1 key
                input = e[layer_name]["input"]
                output = e[layer_name]["output"]
                scale = e[layer_name]["scale"]
                weight = e[layer_name]["weight"]
                file.write(f"NAME:{layer_name}\n")
                file.write(f"SCALE:{scale}\n")
                file.write(f"IN_SHAPE:{input.size()}\n")
                file.write(f"OUT_SHAPE:{output.size()}\n")
                file.write(f"WEIGHT_SHAPE:{weight.size()}\n")

                file.write("INPUT\n")
                for c in input.flatten():  # NCHW format
                    file.write(f"{int(c)}\n")
                file.write("OUTPUT\n")
                for c in output.flatten():  # NCHW format
                    file.write(f"{int(c)}\n")
                file.write("WEIGHT\n")
                for c in weight.flatten():  # NCHW format
                    file.write(f"{int(c)}\n")
        
        