



#region DianaActivationFuser
#### Fuser ### 
# there will always be an output quantizer for analog conv. #fuse div clips than onlu have 1 out and their bitwidth is the same (scale of 1 of 2nd activations)
# Added option for branch fusing 
from typing import List
from DianaModules.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from quantlib.algorithms.qmodules.qmodules.qmodules import _QActivation
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
import torch 

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
                predecessors = [ p for p in node.all_input_nodes ] 
                users = [u for u in node.users]
                if len(predecessors) == 1 and predecessors[0].target in names:# and  issubclass(type(g.get_submodule(users[0].target)), _QActivation): 
                    aps.append(DianaAps('' , node))
               
        
        return aps 
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 
class DianaQuantizerFuserApplier(Applier) : 
    def __init__(self):
        super().__init__() 
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        node = ap.node
        pre_node = [p for p in ap.node.all_input_nodes][0]
        pre_module = g.get_submodule(pre_node.target) 
        cur_module = g.get_submodule(ap.node.target) 
        lower_bitwidth = 0 if (min(pre_module.n_levels, cur_module.n_levels) == pre_module.n_levels ) else 1
        
        pre_node_users = [p for p in pre_node.users]  # upstream
        harmonise_add_nodes= [p for p in pre_node.users if 'AddTreeHarmoniser' in p.target] 
  
        if lower_bitwidth == 0 or len(pre_node_users)-1 == len(harmonise_add_nodes): 
            #if it's 0 then remove cur node 
            cur_node_users = [u for u in ap.node.users ]

            for p_node in cur_node_users: 
                p_node.replace_input_with(ap.node, pre_node)
            # lower bitwidth but zero might be different 
            if(pre_module.zero != cur_module.zero)  : 
                pre_module.zero = max(pre_module.zero , cur_module.zero)  
                pre_module.n_levels =( min(pre_module.zero+pre_module.n_levels,cur_module.zero+cur_module.n_levels) - pre_module.zero)/pre_module.step# (min clip_hi - zero )/step

            g.delete_submodule(ap.node.target)
            g.graph.erase_node(ap.node)
            # Edit clipping of pre_node        

        return g 
#this class is used for fusing activations in true to fake 
class DianaQuantizerFuser(Rewriter) :
    def __init__(self , branch_fuse= True):
        super().__init__("DianaActivationFuser", quantlib_symbolic_trace, DianaQuantizerFuserFinder(), DianaQuantizerFuserApplier()) 
     
#endregion
class DianaBranchFuserFinder(Finder) : 
    def __init__(self, branch_fuse = True) -> None:
        super().__init__()
        self.branch_fuse = branch_fuse
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
            users = [ u for u in node.users]
            act_users = [u for u in users if u.target in names]  
            if len(act_users) == len(users) and len(users)!= 0  : 
                aps.append(DianaAps("branch", node))
        
        return aps 
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 
class DianaBranchFuserApplier(Applier) : 
    def __init__(self):
        super().__init__() 
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        pred = ap.node 
        acts = [u for u in ap.node.users]  
        act_mods= [g.get_submodule(u.target) for u in acts]
        #users = {}
        #for act in acts:  
        #    for u in act.users: 
        #        users.add(u)
            
        def min_n( modules): 
            temp = torch.tensor([1000000])
            index = 0 
            clip_lo = torch.Tensor([-1e10 ])
            clip_hi = torch.Tensor([1e10])
            # this assumes steps of 1 but should work for binary case as well 
            for idx in range(len(modules)): 
                mod = modules[idx]
                if mod.n_levels < temp:   
                    index = idx  
                    temp = mod.n_levels
                clip_lo = torch.max(clip_lo , mod.zero)
                clip_hi = torch.min(clip_hi , mod.zero+mod.n_levels) 

            return index , clip_lo , clip_hi
        # find module with the lowest n_levels ,clip_lo, clip_hi
        lowest_index ,clip_lo,clip_hi = min_n(act_mods)
        removed_nodes= [act for i ,act in enumerate(acts) if i!=lowest_index]
        print(f"Lowest index: {lowest_index} , len of acts: {len(acts)} , len of modules: {len(act_mods)}")
        main_node = acts[lowest_index]
        main_mod = act_mods[lowest_index]    
        # clip main module to the clipping of lowest found activation 
        if main_mod.zero != clip_lo: 
            main_mod.zero = clip_lo 
            main_mod.n_levels =(clip_hi - clip_lo)/main_mod.step

        # replace uses of deleted acts with main_mod
        # AND remove delete acts from graph and delete moduls
        for i in range(len(removed_nodes)): 
            node = removed_nodes[i] 
            node.replace_all_uses_with(main_node)
            g.delete_submodule(node.target)
            g.graph.erase_node(node)
        return g 

class DianaBranchFuser(Rewriter): 
    def __init__(self):
        super().__init__("DianaBranchFuser", quantlib_symbolic_trace, DianaBranchFuserFinder(), DianaBranchFuserApplier())