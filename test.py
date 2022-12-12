
# gmodule.QIdentityEpsRequantiser[1][NOUMBER].div -> gmodule.QIdentityEpsRequantiser[1].div 
from collections import OrderedDict


def load_extra (old_state_dict):
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            name = k
            if "Requantiser" in k: 
                index = [i for i, x in enumerate(k) if x == "["][1]
                name = k[:index] + k[6+index:]
            new_state_dict[name] = v
    
        return new_state_dict

#print(load_extra({"gmodule.QIdentityEpsRequantiser[1][6331].div": "1"}))