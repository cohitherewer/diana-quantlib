r"""Neural network layers are mapped to standard pytorch models in preparation for model export noise nodes are moved """
from DianaModules.utils.grapheditors.layer_integrization.AnalogCoreOperation import AnalogConvIntegrizer
from DianaModules.utils.grapheditors.layer_integrization.LinearOpQuant import DianaLinearOpIntegrizer


from quantlib.editing.editing.editors.base.composededitor import ComposedEditor

import torch.fx as fx
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import EpsTunnelRemover


class LayerIntegrizationConverter(ComposedEditor) :
    def __init__(self):
        super().__init__([AnalogConvIntegrizer(), DianaLinearOpIntegrizer() , EpsTunnelRemover()]) 


        
        