from yaml import YAMLObject
from typing import Dict, Tuple, Union

# for seperable convs , add option to divide channels 
CoreDescriptor = Union[str, Dict[str, Union[Tuple ,int]]]

class ModuleDescriptor(YAMLObject):  
    yaml_tag = u'!ModuleDescriptor' 
    def __init__(self , name :str , core_choice :   CoreDescriptor, qhinitparam: str = 'meanstd', qgranularityspec: Union[str , None] = None) -> None:
        self.name : str = name# Module name in nn Model (use node name 
        self.core_choice : str  = core_choice# ,Analog, Digital or {'Analog': (0 ,31) , 'Digital': 11 } (depth-wise seperable convs )  # default 
        self.qhinitparam : str = qhinitparam# minmax , const , meanstd(mean + 3std)
        self.qgranularityspec :Union[str , None]  = qgranularityspec# per-array , per-outchannel_weights , #
    def __repr__(self):
        return "%s(name=%r, core_choice=%r, qhinitparam=%r, qgranularityspec=%r)" % (self.__class__.__name__, self.name, self.core_choice, self.qhinitparam, self.qgranularityspec) 
    def get_dict(self) : 
        return {self.name: {'core_choice':self.core_choice , 'qhinitparam': self.qhinitparam, 'qgranularityspec': self.qgranularityspec}}
