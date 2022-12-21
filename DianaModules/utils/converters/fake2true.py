r"""Neural network layers are mapped to standard pytorch models in preparation for model export noise nodes are moved """
from DianaModules.utils.grapheditors.hw_mapping.Requantizer import (
    DianaRequantizer,
)
from DianaModules.utils.grapheditors.layer_integrization.AnalogCoreOperation import (
    AnalogConvIntegrizer,
)
from DianaModules.utils.grapheditors.layer_integrization.AnalogNoiseDisabler import (
    AnalogNoiseDisabler,
)
from DianaModules.utils.grapheditors.layer_integrization.AnalogOutRequant import (
    AnalogOutRequantizer,
)
from DianaModules.utils.grapheditors.layer_integrization.LinearOpQuant import (
    DianaLinearOpIntegrizer,
)


from quantlib.editing.editing.editors.base.composededitor import ComposedEditor

import torch.fx as fx
from quantlib.editing.editing.fake2true.epstunnels.inserter.rewriter import (
    EpsTunnelInserter,
)
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import (
    EpsTunnelRemover,
)
from quantlib.editing.editing.fake2true.epstunnels.simplifier.rewriter import (
    EpsTunnelConstructSimplifier,
)
from quantlib.editing.graphs.fx.tracing import QuantLibTracer

# ! Important: If model's layer size is very large and model is trained with partial sums awareness,
# ! the result after layer integration cannot be used. The model will have to be tested on the actual chip
class LayerIntegrizationConverter(ComposedEditor):
    def __init__(self):
        super().__init__(
            [
                AnalogNoiseDisabler(),  # removed by onnx-runtime in exported onnx
                AnalogConvIntegrizer(),
                DianaLinearOpIntegrizer(),  # problem is hereee
                AnalogOutRequantizer(),
                EpsTunnelConstructSimplifier(),
                EpsTunnelRemover(),
            ]
        )

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        for editor in self._children_editors:
            g = editor(g)

        return g
