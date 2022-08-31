

from pathlib import Path 
import torchvision 
import torchvision.datasets as ds  
import torch
from DianaModules.core.Operations import AnalogConv2d, DIANAReLU, DianaBaseOperation 
from DianaModules.models.cifar10.LargeResnet import resnet20
from DianaModules.utils.BaseModules import DianaModule
from torch import nn 
import torch.utils.data as ut
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.optimisers import PACTSGD
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule

from quantlib.editing.editing.tests import ILSVRC12 

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
train_scale = torch.Tensor([2**-8]) 
validation_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

output_weights_path = str(Path("zoo/cifar10/paper/resnet20").absolute() ) 
FP_weight = str(Path("zoo/cifar10/paper/resnet20/ResNet_FPweights.pth").absolute() ) 
model = resnet20() # FP model 
model.load_state_dict(torch.load(FP_weight)['state_dict'])
model.eval()
data_loader = {'train': ut.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True) , 'validate' : ut.DataLoader(validation_dataset, batch_size=128, shuffle=True  ,pin_memory=True)}
#print("model_acc is : " , torch.load(FP_weight)['acc'])
#d_model = DianaModule(model)
#d_model.attach_train_dataset(train_dataset , train_scale) 
#d_model.attach_validation_dataset(validation_dataset , train_scale) 
#print("Floating point model acc: ", d_model.evaluate_model()[1])
#training of FP model1
#d_model.QA_iterative_train(epochs=120, batch_size=128,train_8bit_model=False, train_HW_model=False , train_HWmapped_model=False, output_weights_path=output_weights_path) 


# fake quanitze to 8 bits and initialize quantization then train 
#_8bit_model = DianaModule(DianaModule.from_trained_model(model , map_to_analog=False) ) # 8 bits only (digital core only )
#_8bit_model.attach_train_dataset(train_dataset , train_scale) 
#_8bit_model.attach_validation_dataset(validation_dataset , train_scale) 
    ####for _ , module in  _8bit_model.named_modules(): 
    ####    print (module)
    ###    #
    ###    #
    ###      
#_8bit_model.initialize_quantization()

    ###_8bit_model.gmodule.to(DianaModule.device)
#out_path = output_weights_path + "/"+'ResNet_FQ8weights.pth'
#_8bit_model.gmodule.load_state_dict(torch.load(out_path)['state_dict']) #change names of rewriters to have more standard names
    ####region train 8_bit
    ####print("training 8 bit model") 
    ####optimizer = torch.optim.SGD(_8bit_model.gmodule.parameters(), lr = 0.01 , momentum=0.9,  weight_decay=1e-5)
    ###    #
    ####_8b_metrics =  DianaModule.train(_8bit_model.gmodule,optimizer,data_loader, epochs=24, model_save_path=out_path ) 
    ####train end 
    ###_8bit_model.clip_scales_pow2()
    ###    #
#_8bit_model.clip_scales_pow2()
#print("fake quantized 8bit model acc: ", _8bit_model.evaluate_model()[1])
    ####true quantization , validate accuracy 
#_8bit_model.gmodule = _8bit_model.gmodule.to('cpu')
#_8bit_model.true_quantize()#ILSVRC12.ResNet.RNHeadRewriter()) 
    ###
    #### for debugging (do later)
    ####names = []
    ###for _ , mod in _8bit_model.named_modules(): 
    ###    print(mod) 
    ####for node in _8bit_model.gmodule.graph.nodes: 
    ####    if node.target in names : 
    ####        predecessor = [p for p in node.all_input_nodes]
    ####        users = [u for u in node.users] 
    ####        print(f'For {node} has predecessors ', *predecessor , " and users : " ,*users ) 
#print("true quantized 8bit model acc: ", _8bit_model.evaluate_model()[1])
# end 

## Fake quantize to mixed model 

_Mixed_model = DianaModule(DianaModule.from_trained_model(model , map_to_analog=True ) ) # 8 bits only (digital core only )
#for _ , module in _Mixed_model.named_modules(): 
#    print(module )
print("finished conversion") 
_Mixed_model.attach_train_dataset(train_dataset , train_scale) 
_Mixed_model.attach_validation_dataset(validation_dataset , train_scale) 
_Mixed_model.gmodule.to(DianaModule.device)
_Mixed_model.initialize_quantization()
_Mixed_model.map_scales(HW_Behaviour=True)
#for _ , module in _Mixed_model.named_modules()  : 
#    if isinstance(module, DIANAReLU) : 
#        print("min_flor is: ", module.min_float , "max_float: " , module.max_float, " scale:  " , module.scale)  
print("finished initialization")
##train 
_Mixed_model.gmodule.to(DianaModule.device)
out_path_m = output_weights_path + "/"+'ResNet_Mixedweights.pth'
out_path = output_weights_path + "/"+'ResNet_MixedBestweights.pth'
_Mixed_model.gmodule.load_state_dict(torch.load(out_path_m)['state_dict']) # out_path previously optimized by bound 87.24 validation


#_Mixed_model.clip_scales_pow2()

#optimizer_p = PACTSGD(_Mixed_model.gmodule , lr=0.0004 , pact_decay = 1e-6)
#for _ ,module in _Mixed_model.gmodule.named_modules():
#    if isinstance(module,DIANAReLU) : 
#        module.freeze()
#optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters(), lr = 0.01, weight_decay=5e-5)#0.9114
#best_acc = 0 
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=60, model_save_path=out_path_m ) 
###for e in range(5): 
###    params =  DianaModule.train(_Mixed_model.gmodule,optimizer_p,data_loader, epochs=24, model_save_path=out_path_m ) 
###    if params['validate']['acc'][0] > best_acc: 
###        torch.save ({
###                        'epoch': e,
###                        'state_dict': _Mixed_model.gmodule.state_dict(),
###                        'loss': params['validate']['loss'][0],
###                        'acc' : params['validate']['acc'][0] 
###                    } , out_path)
###        best_acc = params['validate']['acc'][0]      
###
###    for _ ,module in _Mixed_model.gmodule.named_modules():
###        if isinstance(module,DIANAReLU) : 
###            module.freeze()
###    print("Finished PACT SGD")
###    params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=24, model_save_path=out_path_m ) 
###    if params['validate']['acc'][0] > best_acc: 
###        torch.save ({
###                        'epoch': e,
###                        'state_dict': _Mixed_model.gmodule.state_dict(),
###                        'loss': params['validate']['loss'][0],
###                        'acc' : params['validate']['acc'][0] 
###                    } , out_path)
###        best_acc = params['validate']['acc'][0]    
###    for _ ,module in _Mixed_model.gmodule.named_modules():
###        if isinstance(module,DIANAReLU) : 
###            module.thaw()
###    
###    pass 



##train end 

print("fake quantized mixed model acc: ", _Mixed_model.evaluate_model()[1])
#true quantization , validate accuracy 
_Mixed_model.gmodule = _Mixed_model.gmodule.to('cpu')
_Mixed_model.map_to_hw() 
_Mixed_model.integrize_layers()#[ILSVRC12.ResNet.RNHeadRewriter()]) 
_Mixed_model.gmodule.to(DianaModule.device)
print("finished integrization") 

#for  name , module in _Mixed_model.named_modules(): 
#    print(module ) 
for node in _Mixed_model.gmodule.graph.nodes: 
    try: 
       if isinstance(_Mixed_model.gmodule.get_submodule(node.target) , DianaBaseOperation) or node.target in ('view', 'avgpool') : 
            predecessors = [p for p in node.all_input_nodes] 
            users = [u  for u  in node.users]  
            print(f'Diana Node {node} of type {type(_Mixed_model.gmodule.get_submodule(node.target))} with predecessors: ' , *predecessors , " and users: " , *users) 


            #user = users[0]
            #for i in range(6): 
            #    users = [u  for u  in user.users]  
            #    print(f'for analogconv2d user : {user} has type: {type(_Mixed_model.gmodule.get_submodule(user.target))} ' , " has users: " , *users) 
            #    user = users[0]

    except: 
        continue 
         
print("true quantized mixed model acc: ", _Mixed_model.evaluate_model()[1])


# converted_model  = DianaModule.from_trained_model(model , map_to_analog=True) # 8 bits only (digital core only )
