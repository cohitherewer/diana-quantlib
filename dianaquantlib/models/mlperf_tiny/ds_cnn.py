import torch
from torch import nn


class DSCNN(nn.Module):

    def __init__(self):
        super().__init__()

        num_filters = 64

        modules = [
            nn.Conv2d(1, num_filters, (10, 4), (2, 2), (5, 1), bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            #nn.Dropout2d(0.2)
        ]

        for i in range(4):
            modules += [
                nn.Conv2d(num_filters, num_filters, 3, 1, 1, groups=num_filters, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 1, 1, 0, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ]

        modules += [
            nn.Dropout(0.4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, 12)
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def get_random_input(self):
        return torch.randn(1, 1, 10, 49)


if __name__ == '__main__':
    m = DSCNN()
    print(m)
    x = m.get_random_input()
    y = m(x)
    print(y.shape)
