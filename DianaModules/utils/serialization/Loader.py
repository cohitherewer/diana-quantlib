from torch import nn 
import yaml 
from DianaModules.utils.converters.float2fake import F2FConverter
from DianaModules.utils.serialization.Descriptor import ModuleDescriptor
def unknown(loader, suffix, node): 
    return None

class ModulesLoader:  
    def __init__(self) -> None:
        pass
    def load(self, path : str) : 
        descriptors = None

        
        with open(path, 'r') as f: 
            yaml.add_multi_constructor('!', lambda loader, suffix, node: None)
            yaml.add_multi_constructor('', lambda loader, suffix, node: None)
            descriptors= [descriptor.get_dict() for descriptor in list(yaml.load_all(f, yaml.FullLoader) )]
            descriptors = {k: v for d in descriptors for k, v in d.items()}
        return descriptors  # returns 
