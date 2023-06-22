import torch
from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = nn.Identity()
        if stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.skip(x) + self.res(x))


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = 12

        modules = [
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 16, 1),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 64, 2),
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
    m = ResNet()
    print(m)
    x = m.get_random_input()
    y = m(x)
    print(y.shape)
