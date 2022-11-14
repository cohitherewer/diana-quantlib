import torch 

a = torch.load ("zoo/imagenet/resnet18/quantized/weights.pth", map_location="cpu") 
print (a) 