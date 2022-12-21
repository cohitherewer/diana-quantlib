from quantlib.algorithms.qalgorithms import register
from DianaModules.core.Operations import (
    DIANAConv2d,
    DIANAIdentity,
    DIANALinear,
    DIANAReLU,
)
from torch import nn

from quantlib.algorithms.qalgorithms.modulemapping.modulemapping import (
    ModuleMapping,
)

NNMODULE_TO_DIANA = ModuleMapping(
    [
        (nn.Identity, DIANAIdentity),
        (nn.ReLU, DIANAReLU),
        (nn.Linear, DIANALinear),
        (nn.Conv2d, DIANAConv2d),
    ]
)

register["DIANA"] = NNMODULE_TO_DIANA
# import onnx
# import torch
# def symbolic_python_op(ctx: torch.onnx.SymbolicContext, g: torch._C.Graph, *args, **kwargs):
# n = ctx.cur_node


# name = kwargs["name"]
# ret = None
# if name == "DigitalQuantOp":

# ret =g.op("Clip" ,g.op("Floor", g.op("Div", args[0] , args[1] )) , args[2] , args[3])


# else:
## Logs a warning and returns None
# return torch._unimplemented("prim::PythonOp", "unknown node kind: " + name)
## Copy type and shape from original node.


# ret.setType(n.output().type())
# return ret

# from torch.onnx import register_custom_op_symbolic
# register_custom_op_symbolic("prim::PythonOp", symbolic_python_op, 1)
