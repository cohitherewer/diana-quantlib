

from pathlib import Path
import sched 
import torchvision 
import torchvision.datasets as ds  
import torch
from DianaModules.core.Operations import AnalogConv2d, DIANAReLU, DianaBaseOperation 
from DianaModules.models.cifar10.LargeResnet import resnet20, test
from DianaModules.utils.BaseModules import DianaModule
from torch import nn 
import torch.utils.data as ut
from quantlib.algorithms.qalgorithms.qatalgorithms.pact.optimisers import PACTSGD, PACTAdam
from quantlib.algorithms.qmodules.qmodules.qmodules import _QActivation, _QModule

from quantlib.editing.editing.tests import ILSVRC12
from torch.backends.cudnn import benchmark
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel 

train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#train_scale = torch.Tensor([0.0354]) # meanstd
train_scale = torch.Tensor([0.03125]) # meanstd
#train_scale = torch.Tensor([0.0217]) # minmax
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

output_weights_path = str(Path("zoo/cifar10/resnet20").absolute() ) 
FP_weight = str(Path("zoo/cifar10/resnet20/ResNet_FPweights.pth").absolute() ) 
model = resnet20() # FP model 
#model = nn.DataParallel(model, device_ids=[0 ,  1]).cuda() 
model.load_state_dict(DianaModule.remove_data_parallel(torch.load(FP_weight)['state_dict']))
#model.eval()
data_loader = {'train': ut.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True , num_workers=4) , 'validate' : ut.DataLoader(test_dataset, batch_size=128, shuffle=True  ,pin_memory=True, num_workers=4)}
#print("FP _ model_acc is : " , torch.load(FP_weight)['acc'])
#d_model = DianaModule(model) 
#d_model.gmodule = nn.DataParallel(d_model.gmodule, device_ids=[0 ,  1]).cuda() 
#d_model.gmodule = d_model.gmodule.to(DianaModule.device) 
#d_model.attach_train_dataset(train_dataset , train_scale) 
#d_model.attach_validation_dataset(test_dataset, train_scale) 

#print("Floating point model acc: ", d_model.evaluate_model()[1])

#training of FP model1
#print("TRAINING FP MODEL") 
#optimizer = torch.optim.SGD(d_model.gmodule.parameters(),lr = 0.001 ,momentum=0.9, weight_decay=5e-5)
#_ = DianaModule.train(d_model.gmodule, optimizer,data_loader, epochs=120, model_save_path=FP_weight )


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
print("finished conversion") 
_Mixed_model.attach_train_dataset(train_dataset , train_scale) 
_Mixed_model.attach_validation_dataset(test_dataset , train_scale) 
#_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule,device_ids=[0 , 1]).cuda()
_Mixed_model.gmodule = _Mixed_model.gmodule.to(DianaModule.device)
#print("PRINTING FAKE QUANTIZED MODEL")
#for _ , module in _Mixed_model.named_modules(): 
 #   print(module )
#print("NOT PRINTING FAKE QUANTIZED MODEL")


_Mixed_model.initialize_quantization_no_activation(1)
_Mixed_model.map_scales(HW_Behaviour=True)
#for _ , module in _Mixed_model.named_modules()  : 
#    if isinstance(module, DIANAReLU) : 
#        print("min_flor is: ", module.min_float , "max_float: " , module.max_float, " scale:  " , module.scale)  
#print("finished linear layer initialization")
#print("FQ_model Linear Layers Quantized Before Training", _Mixed_model.evaluate_model()[1])
##train 

out_path_FQ_nogain_perarray = output_weights_path + "/"+'ResNet_LinearLayerQuantized_nogain_perarray.pth'
out_path_FQ_nogain_perchannel= output_weights_path + "/"+'ResNet_LinearLayerQuantized_nogain_perchannel.pth'# for conv layers
out_path_FQ_gain_perarray = output_weights_path + "/"+'ResNet_LinearLayerQuantized_gain_perarray.pth'
#out_path_FQ = output_weights_path + "/"+'ResNet_LinearLayerQuantized.pth'
out_path_FQ_A = output_weights_path + "/"+'ResNet_ActQuantized.pth'
out_path_FQ_A_test= output_weights_path + "/"+'ResNet_ActQuantizedTest.pth'
#out_path = output_weights_path + "/"+'ResNet_MixedBestweights.pth'
#_Mixed_model.gmodule.load_state_dict(torch.load(out_path_FQ_nogain_perchannel, map_location=torch.device('cuda:0'))['state_dict']) # nogain_perarray best acc stored in gain _perarray 
#_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(out_path_FQ_nogain_perchannel, map_location=torch.device('cuda:0'))['state_dict']) )# nogain_perarray best acc stored in gain _perarray 

# 

#_Mixed_model.clip_scales_pow2()

#for _ ,module in _Mixed_model.gmodule.named_modules():
#    if isinstance(module,DIANAReLU) : 
#        module.freeze()
benchmark= True
print("FQ_model Linear Layers Quantized Before Training  ", _Mixed_model.evaluate_model()[1])
print("Starting FQ training Linear Layer")
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters(), lr = 0.0001 ,momentum = 0.1 ,weight_decay=5e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1)# stepping on val accuracy 
current_acc= torch.load(out_path_FQ_nogain_perchannel)['acc']
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=120, model_save_path=out_path_FQ_nogain_perchannel , scheduler = scheduler, current_acc= current_acc ) # training with scale of layer before relu clipped to 2 
print("FQ_model Linear Layers Quantized After Training", _Mixed_model.evaluate_model()[1])

print("initializing activations") 
_Mixed_model.initialize_quantization_activations(1)
_Mixed_model.map_scales(HW_Behaviour=True)
print("activations initialized") 
#_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule).cuda()
#_Mixed_model.gmodule.load_state_dict(torch.load(out_path_FQ_A, map_location=torch.device('cuda:0'))['state_dict']) # for training


#_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(out_path_FQ_A )['state_dict'])  )
_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(out_path_FQ_A_test )['state_dict'])  )
_Mixed_model.gmodule.to(DianaModule.device)
_Mixed_model.clip_scales_pow2()
#print("FQ_model Activations Quantized Before Training", _Mixed_model.evaluate_model()[1])

#train 
#print("Training activations")
optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters(), lr = 0.0001, momentum =0.1 , weight_decay=5e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1)# stepping on val accuracy 
current_acc= torch.load(out_path_FQ_A_test)['acc']
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=120, model_save_path=out_path_FQ_A_test , scheduler = scheduler , current_acc=current_acc) 
# train end


print("FQ_model Activations Quantized After Training", _Mixed_model.evaluate_model()[1])

##train end 

#for _ , mod in _Mixed_model.named_modules(): 
    
#rint("evaluating fake quantized mixed model acc") 
#print("fake quantized mixed model acc: ", _Mixed_model.evaluate_model()[1])
#true quantization , validate accuracy 
out_path_HWmapped= output_weights_path + "/"+'ResNet_HWMapped.pth'# 0.8791
out_path_HWmapped_noise= output_weights_path + "/"+'ResNet_HWMappedNoise.pth'
_Mixed_model.gmodule = _Mixed_model.gmodule.to('cpu')
_Mixed_model.map_to_hw()#[ILSVRC12.ResNet.RNHeadRewriter()]) 


#_Mixed_model.gmodule.load_state_dict(DianaModule.remove_data_parallel(torch.load(out_path_HWmapped_noise )['state_dict']))
#_Mixed_model.gmodule.eval() 

#_Mixed_model.gmodule = nn.DataParallel(_Mixed_model.gmodule,device_ids=[0 , 1]).cuda()
_Mixed_model.gmodule.to(DianaModule.device)
print("evaluating HW mapped quantized mixed model acc") 
print(" HW mapped quantized before training mixed model acc: ", _Mixed_model.evaluate_model()[1])
#train hw-mapped 
print("Training HW mapped quantized model ")
#optimizer = torch.optim.SGD(_Mixed_model.gmodule.parameters(), lr = 0.01, weight_decay = 1e-5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1 , patience=4)# stepping on val accuracy 
#params =  DianaModule.train(_Mixed_model.gmodule,optimizer,data_loader, epochs=120, model_save_path=out_path_HWmapped_noise , integrized = True , scale =train_scale.to(DianaModule.device), scheduler=scheduler ) 
#train end 
#print(" HW mapped quantized after training mixed model acc: ", _Mixed_model.evaluate_model()[1])
_Mixed_model.gmodule.to('cpu')
_Mixed_model.integrize_layers() 
### LOOKING for BN 
##for node in _Mixed_model.gmodule.graph.nodes: 
##    if node.target in bn_layer_names.keys(): 
##        p = [p for p in node.all_input_nodes]
##        p_acc = [p for p in p[0].all_input_nodes] 
##        eps : EpsTunnel = _Mixed_model.gmodule.get_submodule(p_acc[0].target)
##
##        print(f"BN node {node.target} has predecessors ", *p , f" of type {type(_Mixed_model.gmodule.get_submodule(p[0].target))} ", f" which has predecessor epstunnel with eps_in {eps._eps_in}, and eps_out{eps._eps_out} ")
### end 


_Mixed_model.gmodule.to(DianaModule.device)
print("finished integrization") 
counter = 0
eps_tunnel_names = {}
for  name , module in _Mixed_model.named_modules(): 
    if isinstance(module , EpsTunnel): 
        eps_tunnel_names[name] = module
        counter += 1
print( f"found this amount of eps tunnels: {counter} ")
for node in _Mixed_model.gmodule.graph.nodes: 
    if node.target in eps_tunnel_names.keys(): 
        pred = [ p for p in node.all_input_nodes]
        USERS = [ u for u in node.users]
    #    print(f"This eps tunnel has eps_in {eps_tunnel_names[node.target]._eps_in} and eps_out {eps_tunnel_names[node.target]._eps_out} and predecessors: " , *pred , " and users: " , *USERS)
for node in _Mixed_model.gmodule.graph.nodes: 
    try: 
       if isinstance(_Mixed_model.gmodule.get_submodule(node.target) , DianaBaseOperation): 
            predecessors = [p for p in node.all_input_nodes] 
            users = [u  for u  in node.users]  
            print(f'Diana Node {node} of type {type(_Mixed_model.gmodule.get_submodule(node.target))} with predecessors: ' , *predecessors , " and users: " , *users) 
#            for p in predecessors: 
#                if "add" in str(p): 
#                    pred = [p for p in p.all_input_nodes] 
#                    up  = [ u for u in p.users]
#
#                    print("add has the following input nodes ", * pred ," and users: ", * up)
#                    for tunnel in pred: 
#                        mod = _Mixed_model.gmodule.get_submodule(tunnel.target) 
#                        print(f"tunnel {tunnel} has eps_in {mod._eps_in} and eps_out {mod._eps_out}")
#
            


            #user = users[0]
            #for i in range(6): 
            #    users = [u  for u  in user.users]  
            #    print(f'for analogconv2d user : {user} has type: {type(_Mixed_model.gmodule.get_submodule(user.target))} ' , " has users: " , *users) 
            #    user = users[0]

    except: 
        continue 
         
print("evaluating true quantized mixed model acc") 
print("true quantized mixed model acc: ", _Mixed_model.evaluate_model()[1])
data_folder = Path("backend/cifar10/resnet20")

_Mixed_model.gmodule.to('cpu')
#_Mixed_model.export_model(str(data_folder.absolute()))

# converted_model  = DianaModule.from_trained_model(model , map_to_analog=True) # 8 bits only (digital core only )
