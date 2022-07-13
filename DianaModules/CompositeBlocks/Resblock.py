
import torch 
from torch import nn 
from typing import Tuple
from DianaModules.Digital.DIlayers import DQIdentity
from DianaModules.utils.DianaModule import DianaModule
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from Digital.DIlayers import DQScaleBias
from AnIA.ANAlayers import AQConv2D

class QResblock(DianaModule):# conv bn covn bn + res_add
    def __init__(self , qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 downsampling_stride:                   Tuple[int, ...] = 1,
                 padding:                  int = 1,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 ) : 
        if downsampling_stride > 1 : 
            self.conv1 = AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, stride=downsampling_stride, padding=padding, dilation=dilation, groups=groups)
            self.shortcut = nn.Sequential(AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec,kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels,stride = downsampling_stride, padding=padding),DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=out_channels) )
        else: 
            self.conv1 = AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, stride=1, padding=padding, dilation=dilation, groups=groups)
            self.shortcut = nn.Sequential()    
        self.bn1 = nn.Sequential(DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=out_channels)  , nn.ReLU())
        self.qid1 = DQIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) # quantizing output of bn1 to 7bit 
        self.conv2 = AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, stride=1, padding=padding, dilation=dilation, groups=groups) 
        self.bn2 = DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=out_channels) 


    def forward(self , input : torch.Tensor) : 
        output = self.bn2(self.conv2(self.bn1(self.conv1(input)))) + self.shortcut(input)
        return output 
         

