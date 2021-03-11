import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class ZeroPad(Module):
    # Probably should not be a module, but keeps things nice and consistent :)
    def __init__(self, actual, projection):
        super().__init__()

        self.actual = actual
        self.projection = projection

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # Every dimension has 2 ways to expand, and it starts from the dimension coming forward, this means you have to
        # have an array for a 4 dim tensor with 6 values (first 4 are "left right, top down) then the last 2 values of
        # the array actually pad the filters maps (or can be thought of as "front back")
        # https://pytorch.org/docs/stable/nn.functional.html#pad

        padding = [0] * ((len(residual.shape) - 1) * 2)

        assert self.actual == residual.shape[1], "The residual did not match the dimensions expected"


        # pad the "back" of the filter maps (maybe this isn't the right thing to do)
        padding[-1] = self.projection - self.actual

        x: torch.Tensor = F.pad(residual, pad=padding)
        return x


class ProjectionShortcut(Module):
    def __init__(self, actual, projection, stride=(1,1)):
        super().__init__()

        self.Conv = nn.Conv2d(actual, projection, kernel_size=(1, 1), stride=stride)
        self.BN = nn.BatchNorm2d(projection)
        self.ACT = nn.ReLU(inplace=True)

    def forward(self, residual):

        projection = self.Conv(residual)
        projection = self.BN(projection)
        projection = self.ACT(projection)

        return projection
