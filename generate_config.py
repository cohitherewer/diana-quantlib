from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.serialization.Serializer import ModulesSerializer
from DianaModules.models.cifar10.LargeResnet import resnet20
# define model 
module = resnet20() 
model = DianaModule(DianaModule.from_trained_fp(module)) 
# instantiate serializer  and save path
serializer = ModulesSerializer(model.gmodule)  

# serialize  
filepath =  "serialized_models/resnet20.yaml"
serializer.dump(filepath) 