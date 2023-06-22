r""" Maps scales to hw-specific scales. Batchnorm layers are converted to 8 bits mul and add and activations are requantized. Also, enables noise nodes of the analog core and scaled are clipped to power of 2"""
from typing import List, Tuple
from dianaquantlib.utils.grapheditors.hw_mapping.AnalogNoiseEnabler import (
    AnalogNoiseEnabler,
)
from dianaquantlib.utils.grapheditors.hw_mapping.BNFolding import BNFolder
from dianaquantlib.utils.grapheditors.fake_quantization.LayerQuantization import (
    LayerQuantizer,
)

from dianaquantlib.utils.grapheditors.hw_mapping.Requantizer import (
    DianaRequantizer,
)
from dianaquantlib.utils.grapheditors.hw_mapping.ResAddAnalogCore import (
    ResidualAddsAnalogCoreRewriter,
)
from dianaquantlib.utils.grapheditors.layer_integrization.AnalogCoreOperation import (
    AnalogConvIntegrizer,
)
from dianaquantlib.utils.grapheditors.layer_integrization.LinearOpQuant import (
    DianaLinearOpIntegrizer,
)


from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor

from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.fake2true.annotation import F2TAnnotator
from quantlib.editing.editing.fake2true.annotation.inputdescription import (
    InputDescription,
    InputDescriptionSpecType,
)
from quantlib.editing.editing.fake2true.epstunnels.inserter.rewriter import (
    EpsTunnelInserter,
)
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import (
    EpsTunnelRemover,
)
from quantlib.editing.editing.fake2true.epstunnels.simplifier.rewriter import (
    EpsTunnelConstructSimplifier,
)

import torch.fx as fx

from quantlib.editing.editing.editors.base.composededitor import ComposedEditor


class HWMappingConverter(ComposedEditor):
    def __init__(self, custom_editor: List[Editor] = [],
                 representative_dataset=None, enable_noise=False):
        editors = [
            BNFolder(),
            LayerQuantizer(representative_dataset, False),
            QuantLibRetracer(),
            F2TAnnotator(),
            EpsTunnelInserter(),
            ResidualAddsAnalogCoreRewriter(),
            DianaRequantizer(),
        ]

        editor_post = [
            EpsTunnelConstructSimplifier(),
            AnalogNoiseEnabler(enable_noise),
        ]

        super(HWMappingConverter, self).__init__(
            editors + custom_editor + editor_post
        )

    def apply(
        self,
        g: fx.GraphModule,
        inputdescription: InputDescriptionSpecType = InputDescription(),
        *args,
        **kwargs
    ) -> fx.GraphModule:

        g = self._children_editors[0](g)  # `BNFolder`
        g = self._children_editors[1](g)  # `LayerQuantizer`
        g = self._children_editors[2](g)  # `QuantLibRetracer`
        g = self._children_editors[3](g, inputdescription)  # `F2TAnnotator`

        for editor in self._children_editors[4:]:
            g = editor(g)

        return g
