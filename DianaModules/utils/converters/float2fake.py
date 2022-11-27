
r""" Converts regular torch neural networko to a fake-quantized model  """

from typing import List, Tuple

from DianaModules.utils.grapheditors.fake_quantization.ActFuser import DianaQuantizerFuser , DianaBranchFuser
from DianaModules.utils.grapheditors.fake_quantization.AnalogConv import AnalogConvMapper
from DianaModules.utils.grapheditors.fake_quantization.Interposer import DianaF2FInterposer

from quantlib.editing.editing.editors.base.composededitor import ComposedEditor

from quantlib.editing.editing.editors.retracers import QuantLibRetracer

from quantlib.editing.editing.float2fake.canonicalisation import F2FCanonicaliser

from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.retracer import QuantLibHarmonisedAddRetracer
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.rewriter import AddTreeHarmoniser

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescriptionSpecType

from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.rewriter import ModuleWiseConverter
from quantlib.editing.editing.float2fake.quantisation.qdescription.qdescription import QDescriptionSpecType

import torch.fx as fx

from quantlib.editing.graphs.fx.tracing import QuantLibTracer

class F2FConverter(ComposedEditor):
    """General-purpose converter to map floating-point networks into
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 analogcoredescriptionspec : Tuple[str , ...]  ,
                 addtreeforceoutputeps:       bool = False ,
                 map_to_analog : bool = True  ,  
                 modules_descriptors = None
             ):

        super(F2FConverter, self).__init__([
            F2FCanonicaliser(),
            
            F2FQuantiser(
                modulewisedescriptionspec,
                addtreeqdescriptionspec,
                analogcoredescriptionspec, 
                addtreeforceoutputeps, 
                map_to_analog=map_to_analog ,
                modules_descriptors= modules_descriptors

                
            )
         
        ])


class F2FQuantiser(ComposedEditor):
    """Specific Rewriter for the diana chip """

    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                analogcoredescriptionspec : Tuple [str , ...], 
                 addtreeforceoutputeps:       bool,
                 map_to_analog : bool = True , modules_descriptors = None
                 ):
    
        editors = [QuantLibRetracer(),
            
            ModuleWiseConverter(modulewisedescriptionspec)]  
        if map_to_analog : 
            editors.append(AnalogConvMapper(analogcoredescriptionspec, modules_descriptors))
        editors += [DianaF2FInterposer()  ,  
        
            
            
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            ),
            QuantLibRetracer()  ,
           DianaQuantizerFuser() ,# ignore the harmonise adds 
           DianaBranchFuser()
           ]
       
        super(F2FQuantiser, self).__init__(
            editors                   
    ) 