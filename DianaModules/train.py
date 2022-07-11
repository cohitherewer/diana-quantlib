import torch
from torch import nn
import torch.utils.data as ut
from torch import nn , optim
from torchvision import transforms
from torchvision import datasets as ds

import matplotlib as plt


model =None #initialize your model 
optimizer = optim.SGD(model.parameter(), lr = 0.01, momentum=0.9)

def train(epochs = 1000 , batch_size = 50): 
    train_loader = ut.DataLoader(dataset=train_dataset, batch_size=batch_size) 
   
    FP_losses = []
    q8b_losses = []
    qHW_losses = []
    qSc_losses = []
    train_loss = 0.0 
    #Iteration 1 - FP Training
    model.start_observing()
    for e in range(epochs): 
        train_loss = 0.0 
        for x,y in train_loader: # x is a 4d tensor
            optimizer.zero_grad() 
            yhat = model(x)
            
            loss = criterion(yhat, y) 
            loss.backward() 
            optimizer.step() 
          
            train_loss += loss.item() *x.size(0) 
          
        
        
        valid_loss = validate() 
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss /60000} Validation Loss: {valid_loss/10000} ')
        FP_losses.append((train_loss , valid_loss))
    #Iteration 2 - Fake Quantistion all to 8 bit 
    model.stop_observing() 
    #Iteration 3 - Input HW specific quantisation 

    #Iteration 4 - map scales and train 

    
    #PATH = './weights.pth'
    #torch.save(model.state_dict(), PATH)
    #torch.save(model.state_dict(), PATH)
    return losses  
def validate () :
    valid_loss = 0.0
    model.eval()     
    for x, y in validation_loader:
        target = model(x)
        loss = criterion(target,y)
        valid_loss = loss.item() * x.size(0)
    return valid_loss

train_dataset = ds.FashionMNIST(root="./data" , train = True, download = True, transform=transforms.ToTensor())
validation_dataset = ds.FashionMNIST(root="./data" , train = True, download = False , transform=transforms.ToTensor())


validation_loader = ut.DataLoader(dataset=validation_dataset, batch_size=10) 

criterion = nn.CrossEntropyLoss() # for softmax in the end

losses = train(e = 3, batch_size=100) 

#plotting

for i in range(len(losses)): 
    for y1, y2 in losses[i]:
        plt.plot(i,y1 , color='red', label="train losses") 
        plt.plot(i,y2, color='orange', label = "validation losses")




# Training system without quantisation hyperparameters training algorithms 
# Train the network with full precision floating point numbers but have quantisation layers observe range of inputs 

# Iteration 1: Initialise quantisation hyperparameters and re train with fake-quantised model (all quantizations to 8-bit ) 
# Iteration 2: Re-Initialise Quantisation hyperparameters to match with HW constraints
# Iteration 3: Train with HW specificities (power of two scales, and whatever)remove nodes and test directly with quantised weights 
# Compare with HW model 