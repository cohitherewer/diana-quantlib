from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.c1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1
        )
        self.s2 = nn.Conv2d(
            in_channels=6, out_channels=6, kernel_size=2, stride=2
        )
        self.c3 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1
        )
        self.s4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=2, stride=2
        )
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.o = nn.Linear(84, 10)

    def forward(self, x):
        y = F.relu(self.c1(x), inplace=True)
        y = F.relu(self.s2(y), inplace=True)
        y = F.relu(self.c3(y), inplace=True)
        y = F.relu(self.s4(y), inplace=True)
        y = y.reshape((-1, 16 * 5 * 5))
        y = F.relu(self.c5(y), inplace=True)
        y = F.relu(self.f6(y), inplace=True)
        y = self.o(y)
        return y
