import torch.nn as nn
from itertools import chain


class Cppn(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        lst=[nn.Linear(3, 15),
            nn.LeakyReLU(),
            *chain.from_iterable([(nn.Linear(15, 15), nn.LeakyReLU()) for i in range(hidden_layers)] +
                                 [(nn.Linear(15, 3), nn.Sigmoid())])]
        self.activation = nn.Sequential(
            *lst
        )

    def forward(self, input):
        return self.activation(input)
