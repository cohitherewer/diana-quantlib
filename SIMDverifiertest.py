from DianaModules.models.cifar10.LargeResnet import resnet20
import torchvision
import torchvision.datasets as ds 
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.utils.verification.SIMDmodelverifier import SIMDVerifier
from pathlib import Path
import torch 
output_weights_path = str(Path("zoo/cifar10/resnet20").absolute() ) 
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )
train_scale = torch.Tensor([0.03125])

model = resnet20() 

_Mixed_model = DianaModule(DianaModule.from_trainedfp_model(model) ) 

print("converted")


_Mixed_model.attach_train_dataset(train_dataset , train_scale) 
_Mixed_model.attach_validation_dataset(test_dataset , train_scale)
_Mixed_model.initialize_quantization(1) 

_Mixed_model.map_to_hw()#[ILSVRC12.ResNet.RNHeadRewriter()]) 
print("mapped")

out_path_HWmapped= output_weights_path + "/"+'ResNet_HWMapped.pth'
#_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(out_path_HWmapped , map_location=torch.device('cpu'))['state_dict']))

verifier = SIMDVerifier(_Mixed_model.gmodule) 
x,_ =  x, _ =test_dataset.__getitem__(0) 
    
if len(x.shape) == 3 : 
    x = x.unsqueeze(0)

verifier.verify(x) 

# verification process success 
