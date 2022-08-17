
from quantlib.editing.editing.editors.base.rewriter.applicationpoint import ApplicationPoint
from quantlib.editing.graphs import fx

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
