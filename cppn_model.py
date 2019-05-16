import torch.nn as nn
from itertools import chain


class Cppn(nn.Module):
    def __init__(self, hidden_layers, neurons_hidden=15, hidden_function=nn.LeakyReLU, output_function=nn.Sigmoid):
        super().__init__()
        lst = [*chain.from_iterable([
            (nn.Linear(neurons_hidden, neurons_hidden), hidden_function())
            for i in range(hidden_layers)])
        ]
        self.activation = nn.Sequential(
            nn.Linear(3, neurons_hidden),
            hidden_function(),
            *lst,
            nn.Linear(neurons_hidden, 3),
            output_function()
        )

    def forward(self, input):
        return self.activation(input)
