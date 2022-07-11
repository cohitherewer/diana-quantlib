
import torch 
from torch import nn 
from typing import Tuple
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from Digital.DIlayers import DQScaleBias, DQIdentity
from AnIA.ANAlayers import AQConv2D
class QResblock(nn.Module):# conv bn covn bn + res_add
    def __init__(self , qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 downsampling_stride:                   Tuple[int, ...] = 1,
                 padding:                  int = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1,
                 ) : 

        self.conv1 = AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, stride=1, padding=padding, dilation=dilation, groups=groups)
        self.bn1 = DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=out_channels) 
        self.conv2 = AQConv2D(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, stride=downsampling_stride, padding=padding, dilation=dilation, groups=groups) 
        self.bn2 = DQScaleBias(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_channels=out_channels) 

    def forward(self , input : torch.Tensor) : 
        output = self.bn2(self.conv2(self.bn1(self.conv1(input)))) + input # wheen using fake quantiser funciton don't take into account scale
        return output 
         

