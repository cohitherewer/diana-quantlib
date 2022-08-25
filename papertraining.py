

from pathlib import Path 
import torchvision 
import torchvision.datasets as ds  
import torch 
from DianaModules.models.LargeResnet import resnet20
from DianaModules.utils.BaseModules import DianaModule
from torch import nn 
import torch.utils.data as ut
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule

from quantlib.editing.editing.tests import ILSVRC12 

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([1/256]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

output_weights_path = str(Path("zoo/cifar10/paper/resnet20").absolute() ) 
FP_weight = str(Path("zoo/cifar10/paper/resnet20/ResNet_FPweights.pth").absolute() ) 
model = resnet20() # FP model 
model.load_state_dict(torch.load(FP_weight)['state_dict'])
#print("model_acc is : " , torch.load(FP_weight)['acc'])
#d_model = DianaModule(model)
#d_model.attach_train_dataset(train_dataset , train_scale) 
#d_model.attach_validation_dataset(validation_dataset , train_scale) 
#training of FP model1
#d_model.QA_iterative_train(epochs=60, batch_size=128,train_8bit_model=False, train_HW_model=False , train_HWmapped_model=False, output_weights_path=output_weights_path) 


# fake quanitze to 8 bits and initialize quantization then train 
_8bit_model = DianaModule(DianaModule.from_trained_model(model , map_to_analog=False) ) # 8 bits only (digital core only )
_8bit_model.attach_train_dataset(train_dataset , train_scale) 
_8bit_model.attach_validation_dataset(validation_dataset , train_scale) 
#for _ , module in  _8bit_model.named_modules(): 
#    print (module)

data_loader = {'train': ut.DataLoader(_8bit_model.train_dataset['dataset'], batch_size=256, shuffle=True, pin_memory=True) , 'validate' : ut.DataLoader(_8bit_model.validation_dataset['dataset'], batch_size=256, shuffle=True  ,pin_memory=True)}
      
_8bit_model.initialize_quantization()
_8bit_model.clip_scales_pow2()
_8bit_model.gmodule.to(DianaModule.device)
out_path = output_weights_path + "/"+'ResNet_FQ8weights.pth'
_8bit_model.gmodule.load_state_dict(torch.load(out_path)['state_dict']) #change names of rewriters to have more standard names
#region train 8_bit
#print("training 8 bit model") 
#optimizer = torch.optim.SGD(_8bit_model.gmodule.parameters(), lr = 0.04 , momentum=0.9,  weight_decay=1e-5)


#_8b_metrics =  DianaModule.train(_8bit_model.gmodule,optimizer,data_loader, epochs=1, model_save_path=out_path ) 
#train end 
_8bit_model.clip_scales_pow2()

print("fake quantized 8bit model acc: ", _8bit_model.evaluate_model()[1])
#true quantization , validate accuracy 
_8bit_model.gmodule = _8bit_model.gmodule.to('cpu')
_8bit_model.true_quantize()#[ILSVRC12.ResNet.RNHeadRewriter()]) 
names = []
for _ , module in _8bit_model.named_modules(): 
    if isinstance(module, _QModule) :
        names.append(_) 
for node in _8bit_model.gmodule.graph.nodes: 
    if node.target in names : 
        predecessor = [p for p in node.all_input_nodes]
        users = [u for u in node.users] 
        print(f'For {node} has predecessors ', *predecessor , " and users : " ,*users ) 
print("true quantized 8bit model acc: ", _8bit_model.evaluate_model()[1])
# end 

## Fake quantize to mixed model # TODO need to freeze relus upper bound before true quantization 
#_Mixed_model = DianaModule(DianaModule.from_trained_model(model , map_to_analog=True ) ) # 8 bits only (digital core only )
#_Mixed_model.initialize_quantization()
#_Mixed_model.map_scales()
#_Mixed_model.clip_scales_pow2() 
##train 
#
##train end 
#
#
#converted_model  = DianaModule.from_trained_model(model , map_to_analog=True) # 8 bits only (digital core only )
#