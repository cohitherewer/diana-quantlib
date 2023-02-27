import torch
from torch import nn


class MobileNet(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = 4
        modules = [
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        ]

        # spec = (in_channel, out_channel, stride)
        spec = [(8, 16, 1),
                (16, 32, 2),
                (32, 32, 1),
                (32, 64, 2),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 128, 1),
                (128, 128, 1),
                (128, 128, 1),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1)
        ]

        for in_channels, out_channels, stride in spec:
            modules += [
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]

        modules += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def get_random_input(self):
        return torch.randn(1, 3, 96, 96)


if __name__ == '__main__':
    m = MobileNet()
    print(m)
    x = m.get_random_input()
    y = m(x)
    print(y.shape)
