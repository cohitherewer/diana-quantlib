import torch
from torch import nn


class DAE(nn.Module):

    def __init__(self):
        super().__init__()

        num_outputs = 640
        num_units = 128

        unit_spec = [
            (num_outputs, num_units),
            (num_units, num_units),
            (num_units, num_units),
            (num_units, num_units),
            (num_units, 8),
            (8, num_units),
            (num_units, num_units),
            (num_units, num_units),
            (num_units, num_units),
        ]

        modules = []
        for in_units, out_units in unit_spec:
            modules += [
                nn.Linear(in_units, out_units),
                nn.ReLU(),
            ]

        modules += [
            nn.Linear(num_units, num_outputs),
        ]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def get_random_input(self):
        return torch.randn(1, 640)



if __name__ == '__main__':
    m = DAE()
    print(m)
    x = m.get_random_input()
    y = m(x)
    print(y.shape)
