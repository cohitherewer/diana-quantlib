



from pathlib import Path
from typing import Optional
from quantlib.backends.base.onnxannotator import ONNXAnnotator
from quantlib.backends.dory.onnxexporter import DORYExporter

import onnxruntime as rt 

import torch 
from torch import nn 
import os 
class DianaAnnotator(ONNXAnnotator) : 
    pass 

class DianaExporter(DORYExporter) : 
    def __init__(self):
        super().__init__()
    def export(self, network: nn.Module, input_shape: torch.Size, path: os.PathLike, name: Optional[str] = None, opset_version: int = 10) -> None:
        super().export(network, input_shape, path, name, opset_version)
        sess_options = rt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # To enable model serialization after graph optimization set this
        data_folder = Path("backend/optimized/optimized_model.onnx")
        sess_options.optimized_model_filepath = str(data_folder.absolute())
        
        session = rt.InferenceSession(str(path) +"/digital_core_test_QL_NOANNOTATION.onnx", sess_options)