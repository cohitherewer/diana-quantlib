from tracemalloc import start
import torch 
from torch import nn

#test = nn.Sequential(nn.Linear(3,3 ), nn.Linear(3, 1 )) 
#for idx ,m in enumerate(test.modules()): 
#    print(idx , m) 
#    if type(m) == nn.Sequential: 
#        print("sequentuel ")
#        continue
#    rnd = torch.rand(3)
#    out = m(rnd)
#    print ("type: ", type(m)) 
#    print("out:L ", out)
# target output size of 5x7
#m = nn.AdaptiveAvgPool2d((5,7))
#input = torch.randn(1, 64, 8, 9)
#output = m(input)
#print(output.size())
## target output size of 7x7 (square)
#m = nn.AdaptiveAvgPool2d(7)
#input = torch.randn(1, 64, 10, 9)
#output = m(input)
## target output size of 10x7
#m = nn.AdaptiveAvgPool2d((None, 7))

#output = m(input)

feature_map_batch_size = 2
feature_map_H = 4 
feature_map_W = 4 
input_channels = 3 

feature_map = torch.floor(torch.rand(feature_map_batch_size  ,  input_channels,feature_map_H ,feature_map_W ) * 10) 




kernel_size = 2
kernel_count = 3 
kernel = torch.floor((torch.rand(kernel_count , input_channels  , kernel_size,kernel_size)-0.5) * 4)
padding = 0 
stride = 1


flattened_feature_map = torch.nn.functional.unfold(feature_map ,kernel_size =kernel_size ).unsqueeze(1) # 
flattened_feature_map = flattened_feature_map.expand(feature_map_batch_size , kernel_count , flattened_feature_map.size(2) , flattened_feature_map.size(3) )
print(flattened_feature_map.shape)
flattened_kernel = kernel.view(kernel.size(0), -1).unsqueeze(0).expand(feature_map_batch_size , kernel_count , -1).unsqueeze(3).expand(feature_map_batch_size,kernel_count, flattened_feature_map.size(2), flattened_feature_map.size(3))

flattened_out = flattened_kernel * flattened_feature_map
out_H = int((feature_map_H-kernel_size+2* padding)/stride+1 ) 
out_W = int((feature_map_W-kernel_size+2* padding)/stride+1 )


#flattened_out = torch.sum(flattened_out , 2).reshape(feature_map_batch_size ,kernel_count, out_H ,out_W )
#negative addition 
negative_mask = flattened_out <0


negative_clip = -10 
positive_clip = 10

negative_out = torch.max(torch.sum(flattened_out*negative_mask , 2)  , negative_clip) 
positive_out = torch.min(torch.sum(flattened_out*~negative_mask , 2), positive_clip) 

total_out = (negative_out + positive_out).reshape(feature_map_batch_size ,kernel_count, out_H ,out_W )



conv_out = nn.functional.conv2d(feature_map , kernel , None , stride  , padding=padding)
print(conv_out)
print(total_out)
print ("SUCCESS" if torch.all (conv_out.eq(total_out)) else "failure")