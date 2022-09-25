from DianaModules.models.cifar10.LargeResnet import resnet20
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.serialization.Loader import ModulesLoader
from DianaModules.utils.serialization.Serializer import ModulesSerializer
from pathlib import Path
model = resnet20() # FP model 

_Mixed_model = DianaModule(DianaModule.from_trainedfp_model(model ) ) # 8 bits only (digital core only )

#pth = Path("kernels/test.yaml")
pth = "/imec/other/csainfra/nada64/DianaTraining/serialized_models/test.yaml"

serializer = ModulesSerializer(_Mixed_model.gmodule)  
serializer.dump(pth) 

loader = ModulesLoader()
converted_model = loader.load(pth)