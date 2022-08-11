from DianaModules.utils.BaseModules import DianaModule 
from DianaModules.models.resnet import resnet20
import torchvision
import torchvision.datasets  as ds 

dataset = ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

resnet = resnet20() 

converted_dmod = DianaModule(resnet) 
converted_dmod.gmodule = DianaModule.from_trained_model(resnet) 

converted_dmod.start_observing() 

for  i in range(100) : 
    x, _ = dataset.__getitem__(i) 
    x = x.unsqueeze(0) 
    _ = converted_dmod(x)

converted_dmod.map_scales(HW_Behaviour=True) 
converted_dmod.stop_observing() 



