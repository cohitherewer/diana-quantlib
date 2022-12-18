import sys
sys.path.insert(1, '/imec/other/csainfra/nada64/DianaTraining/resnet8/fast')
from torch import nn 
import torch 
import torch.nn.functional as F 
channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
def conv_bn(c_in , c_out, bias = False): 
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=bias), nn.BatchNorm2d(c_out)
class resnet8(nn.Module): 
    def __init__(self) -> None:
        super().__init__() 
        self.prep_conv  , self.prep_bn    = conv_bn(3, channels["prep"]) 
        self.layer1_conv, self.layer1_bn  = conv_bn(channels["prep"], channels["layer1"])
        self.layer2_conv, self.layer2_bn  = conv_bn(channels["layer1"], channels["layer2"])
        self.layer3_conv, self.layer3_bn  = conv_bn(channels["layer2"], channels["layer3"])
        # residuals 
        self.layer1_residual_res1_conv , self.layer1_residual_res1_bn = conv_bn(channels["layer1"],channels["layer1"])
        self.layer1_residual_res2_conv , self.layer1_residual_res2_bn = conv_bn(channels["layer1"],channels["layer1"])
        self.layer3_residual_res1_conv , self.layer3_residual_res1_bn = conv_bn(channels["layer3"],channels["layer3"])
        self.layer3_residual_res2_conv , self.layer3_residual_res2_bn = conv_bn(channels["layer3"],channels["layer3"])
        self.pool1 = nn.MaxPool2d(2) 
        self.pool2 = nn.MaxPool2d(2) 
        self.pool3 = nn.MaxPool2d(2) 
        self.globalpool = nn.MaxPool2d(4) 
        self.linear = nn.Linear(channels["layer3"], 10, bias=False)
    def forward(self, x: torch.Tensor): 
        out = F.relu(self.prep_bn(self.prep_conv(x)), inplace=True) 
        out = self.pool1(F.relu(self.layer1_bn(self.layer1_conv(out)), inplace=True) ) 
        shortcut = F.relu(self.layer1_residual_res2_bn(self.layer1_residual_res2_conv(F.relu(self.layer1_residual_res1_bn(self.layer1_residual_res1_conv(out)), True))), True)
        out = out + shortcut 
        out = self.pool2(F.relu(self.layer2_bn(self.layer2_conv(out)), inplace=True) ) 
        out = self.pool3(F.relu(self.layer3_bn(self.layer3_conv(out)), inplace=True) ) 
        shortcut = F.relu(self.layer3_residual_res2_bn(self.layer3_residual_res2_conv(F.relu(self.layer3_residual_res1_bn(self.layer3_residual_res1_conv(out)), True))), True)
        out = out + shortcut 
        out = self.globalpool(out) 
        out = out.view(out.size(0), -1)
        out = self.linear(out) 
        return out