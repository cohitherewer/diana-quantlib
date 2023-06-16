import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class CifarNet8(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = 12

        modules = [
            ConvBlock(3,  16, 1),
            ConvBlock(16, 16, 1),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 32, 1),
            ConvBlock(32, 32, 1),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 64, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def get_random_input(self):
        return torch.randn(1, 3, 32, 32)


if __name__ == '__main__':
    m = CifarNet8()
    print(m)
    x = m.get_random_input()
    y = m(x)
    print(y.shape)
