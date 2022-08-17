from typing import List

from DianaModules.utils.grapheditors.fake_quantization.ActFuser import DianaQuantizerFuser
from DianaModules.utils.grapheditors.fake_quantization.Interposer import DianaF2FInterposer
from DianaModules.utils.grapheditors.true_quantization.LinearOpQuant import DianaLinearOpIntegrizer
from DianaModules.utils.grapheditors.true_quantization.Requantizer import DianaRequantizer


from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor

from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.fake2true.annotation import F2TAnnotator
from quantlib.editing.editing.fake2true.annotation.inputdescription import InputDescription, InputDescriptionSpecType
from quantlib.editing.editing.fake2true.epstunnels.inserter.rewriter import EpsTunnelInserter
from quantlib.editing.editing.fake2true.epstunnels.remover.rewriter import EpsTunnelRemover
from quantlib.editing.editing.fake2true.epstunnels.simplifier.rewriter import EpsTunnelConstructSimplifier


from quantlib.editing.editing.float2fake.canonicalisation import F2FCanonicaliser

from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.retracer import QuantLibHarmonisedAddRetracer
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.rewriter import AddTreeHarmoniser

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescriptionSpecType

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.rewriter import ModuleWiseConverter
from quantlib.editing.editing.float2fake.quantisation.qdescription.qdescription import QDescriptionSpecType

import torch.fx as fx

class DianaF2FConverter(ComposedEditor):
    """General-purpose converter to map floating-point networks into
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
             ):

        super(DianaF2FConverter, self).__init__([
            F2FCanonicaliser(),
            
            DianaF2FQuantiser(
                modulewisedescriptionspec,
                addtreeqdescriptionspec,
                addtreeforceoutputeps,
            )
         
        ])
# this assumes that each module has fake-quantized identity output mapping (NO need for QuantiserInterposer).
class DianaF2FQuantiser(ComposedEditor):
    """Specific Rewriter for the diana chip """

    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 ):
      
        super(DianaF2FQuantiser, self).__init__([
            QuantLibRetracer(),
            
            ModuleWiseConverter(modulewisedescriptionspec),
       
           
            
            DianaF2FInterposer()  ,  
        
            
            
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            ),
            QuantLibRetracer()  ,
            DianaQuantizerFuser() ,# ignore the harmonise adds 
           
           
        ]) 

class DianaF2TConverter(ComposedEditor) : 
    def __init__(self, custom_editor : List[Editor] = []) : 
        editors = [
            
            QuantLibRetracer(),
          
            F2TAnnotator(),
            EpsTunnelInserter(),
         
            DianaLinearOpIntegrizer(), 
         
            DianaRequantizer()]
        
        editor_post = [   
            EpsTunnelConstructSimplifier(),
            EpsTunnelRemover()
        ]

        super(DianaF2TConverter, self).__init__(editors + custom_editor + editor_post)

    def apply(self,
              g: fx.GraphModule,
              inputdescription: InputDescriptionSpecType = InputDescription(),
              *args,
              **kwargs) -> fx.GraphModule:

        g = self._children_editors[0](g)                    # `QuantLibRetracer`
        g = self._children_editors[1](g, inputdescription)  # `F2TAnnotator`
        for editor in self._children_editors[2:]:
            g = editor(g)

        return g


