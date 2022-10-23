from torch import nn 
import yaml 
from DianaModules.utils.converters.float2fake import F2FConverter
from DianaModules.utils.serialization.Descriptor import ModuleDescriptor

class ModulesLoader:  
    def __init__(self) -> None:
        pass
    def load(self, path : str) : 
        descriptors = None
        with open(path, 'r') as f: 
            descriptors= [descriptor.get_dict() for descriptor in list(yaml.full_load_all(f) )]
            descriptors = {k: v for d in descriptors for k, v in d.items()}
        return descriptors  # returns 
