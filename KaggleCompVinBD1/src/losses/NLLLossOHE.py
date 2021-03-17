import torch
from torch import nn


class NLLLossOHE(nn.Module):

    def __init__(self):
        super().__init__()

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, predictions, targets):
        # One hot encoded vectors need some special treatment that native PyTorch modules don't support :(
        return (-self.LogSoftmax(predictions) * targets).sum(1).mean()
