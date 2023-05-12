import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Swish(nn.Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, input):
        return input * torch.sigmoid(self.weight * input)


ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "swish": Swish
}
