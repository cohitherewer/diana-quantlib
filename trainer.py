import argparse
from torch.utils.data import DataLoader , Dataset
from DianaModules.utils.BaseModules import DianaModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from DianaModules.utils.compression.ModelDistiller import QModelDistiller
from DianaModules.utils.compression.QuantStepper import QuantDownStepper

from DianaModules.utils.serialization.Loader import ModulesLoader 

# define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--stage' ,type=str, default='fp' , 
                    help='defines the stage in the conversion process where the model will be trained in')
parser.add_argument('--batch_size', type=int, default=32,
                    help='the batch size for training and validation')
parser.add_argument('--num_workers', type=int, default=4,
                    help='the number of workers for the data loaders')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='the number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='the learning rate for the optimizer') 
parser.add_argument('--momentum', type=float, default=0.0, 
                    help="the momentum of the optimizer")
parser.add_argument('--log_dir', type=str, default='logs/',
                    help='the directory to save logs and checkpoints')
parser.add_argument('--early_stopping_patience', type=int, default=3,
                    help='the number of epochs to wait for improvement before stopping')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                    help='the directory to save checkpoints')  

# quantization arguments 
# define where the floating point directory is 
parser.add_argument('--fp_pth', type=str , default ="", 
                    help="path to the pre-trained floating point module") 
parser.add_argument('--scale',  type=float , default="0.0" ,
                    help="the scale of dataset going through a quantizer with the same quant range as the input layer")  # you can get it by passing it through a quantizer with the same quantization as the input layer. You can chek out the datasetscale.py file 
parser.add_argument('--config_pth',  type=str, default=None ,
                    help="the path to the model's quantization configuration file (yaml file) ")
parser.add_argument('--quantized_pth' , type=str, default="", 
                    help="path to the quantized floating point model") 
parser.add_argument('--fq_pth', type=str, default="" , 
                    help="path to the trained fake quantized model to be used for hw-mapped training")
parser.add_argument('--quant_steps' , type= int , default=0 , 
                    help="steps needed for trainer drop from 8 bits down to the target quantization"
                    ) 
# parse the arguments
args = parser.parse_args()
# define your Pytorch Lightning module
class MyLightningModule(pl.LightningModule):
  ...

# instantiate the module
module = MyLightningModule() 
# define the datasets 

train_dataset = Dataset() 
val_dataset   = Dataset() 

# define the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True) 

# define the logger
logger = CSVLogger(args.log_dir)
 
# define the early stopping callback
early_stopping = EarlyStopping(monitor='val_acc', patience=args.early_stopping_patience)



def train_fp(): 
  model = DianaModule(module)
  # instantiate the trainer
  # define the checkpoint saving callback

  checkpoint = ModelCheckpoint(args.checkpoint_dir, monitor='val_acc', mode='max',filename=f'FP{module.__name__}-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False, save_top_k=1)
  
  trainer = pl.Trainer(max_epochs=args.num_epochs, logger=logger,
                    distributed_backend='ddp', callbacks=[early_stopping, checkpoint])  
  # train the model
  trainer.fit(model, train_dataloader, val_dataloader)
def train_fq(): 
  # load pre-trained floating point weights 
  module.load_state_dict(torch.load(args.fp_pth, map_location="cpu")["state_dict"])
  # load configurations file 
  module_descriptions_pth = args.config_pth
  module_description = None
  if module_descriptions_pth:  
    loader = ModulesLoader()
    module_description = loader.load(module_descriptions_pth) 
  # fake-quantize model and attach scales 
  model = DianaModule(DianaModule.from_trainedfp_model(module ,modules_descriptors=module_description)) 
  model.attach_train_dataloader(train_dataloader, args.scale) 
  model.attach_quantization_dataloader(train_dataloader) 

  # load quantized model 
  model.set_quantized(activations=False) 
  model.load_state_dict(torch.load(args.quantized_pth, map_location="cpu")["state_dict"])
  # Initialize modules needed for training
  distiller = QModelDistiller(student =model , teacher=module, learning_rate=args.learning_rate, momentum=args.momentum,
           max_epochs=args.num_epochs, weight_decay=args.weight_decay, nesterov=False, lr_scheduler="" ,gamma= 0.0,optimizer="SGD")
  stepper = QuantDownStepper(model, args.quant_steps, initial_quant={"bitwidth": 8, "signed" :True}, target_quant="ternary")
  checkpoint = ModelCheckpoint(args.checkpoint_dir, monitor='val_acc', mode='max',
  filename=f'FQ_NOACT_{module.__name__}-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False, save_top_k=1)
  checkpoint_act = ModelCheckpoint(args.checkpoint_dir, monitor='val_acc', mode='max',filename=f'FQ_{module.__name__}-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False, save_top_k=1)
  for i in range(args.quant_steps): 
    trainer = pl.Trainer(max_epochs=args.num_epochs, logger=logger,
                      strategy='dp', callbacks=[early_stopping, checkpoint])  

    trainer.fit(distiller, train_dataloader , val_dataloader) 
    # Training with quantized activations 
    
    trainer = pl.Trainer(max_epochs=args.num_epochs, logger=logger,
                      strategy='dp', callbacks=[early_stopping, checkpoint_act])    
    #quantize activations 
    model.initialize_quantization_activations(trainer)
    #retrain model with quantized activations 
    trainer.fit(distiller, train_dataloader , val_dataloader)  
    # step down quantization 
    stepper.step()

def train_hw(): 
  # load pre-trained floating point weights 
  module.load_state_dict(torch.load(args.fp_pth, map_location="cpu")["state_dict"])
  # load configurations file 
  module_descriptions_pth = args.config_pth
  module_description = None
  if module_descriptions_pth:  
    loader = ModulesLoader()
    module_description = loader.load(module_descriptions_pth) 
  # fake-quantize model and attach scales 
  model = DianaModule(DianaModule.from_trainedfp_model(module ,modules_descriptors=module_description)) 
  # load fq model  
  model.attach_train_dataloader(train_dataloader, args.scale) 
  model.attach_quantization_dataloader(train_dataloader) 
  model.set_quantized() 
  model.load_state_dict(torch.load(args.fq_pth, map_location="cpu")["state_dict"])  
  
  checkpoint = ModelCheckpoint(args.checkpoint_dir, monitor='val_acc', mode='max',filename=f'HW_{module.__name__}-{epoch:02d}-{val_acc:.4f}' ,save_top_k=1, save_on_train_epoch_end=False, save_top_k=1)
  model.map_to_hw() 
  trainer = pl.Trainer(max_epochs=args.num_epochs, logger=logger,
                      strategy='ddp', callbacks=[early_stopping, checkpoint])  
  trainer.fit(model ,train_dataloader, val_dataloader)
   


def main():  
  if args.stage  == 'fp' : 
    train_fp () 
  elif args.stage == 'fq' :    
    train_fq()
  else: 
    train_hw() #basically hardware conversion is just the redefinition of the original scales and incorporation of DIANA's architecture constraints 
 

if __name__ == "__main__" : 
  main() 
