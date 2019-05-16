import torch.nn as nn
from itertools import chain


class Cppn(nn.Module):
    def __init__(self, hidden_layers, neurons_in_hidden=15, hidden_function=nn.LeakyReLU, output_function=nn.Sigmoid):
        super().__init__()
        lst = [nn.Linear(3, neurons_in_hidden),
               hidden_function(),
               *chain.from_iterable([(nn.Linear(neurons_in_hidden, neurons_in_hidden, hidden_function())) for i in range(hidden_layers)] +
                                    [(nn.Linear(neurons_in_hidden, 3), output_function())])]
        self.activation = nn.Sequential(
            *lst
        )

    def forward(self, input):
        return self.activation(input)
