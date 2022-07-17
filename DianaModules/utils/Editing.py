from typing import Union

from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer
from quantlib.editing.editing.float2fake.canonicalisation import F2FCanonicaliser

from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.retracer import QuantLibHarmonisedAddRetracer
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.rewriter import AddTreeHarmoniser

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescriptionSpecType
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.rewriter import ModuleWiseConverter
from quantlib.editing.editing.float2fake.quantisation.qdescription.qdescription import QDescriptionSpecType


class DianaF2FConverter(ComposedEditor):
    """General-purpose converter to map floating-point networks into
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 qinterposerqdescriptionspec: Union[QDescriptionSpecType, None] = None):

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
                 qinterposerqdescriptionspec: Union[QDescriptionSpecType, None] = None):
        if qinterposerqdescriptionspec == None:
            super(DianaF2FQuantiser, self).__init__([
            QuantLibRetracer(),
            ModuleWiseConverter(modulewisedescriptionspec),
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            )
        ])

            


