import torch
from torch import nn
from torch.nn import Module


#TODO - make this for any size of conv blocks, and fix the downsampling issue
# https://arxiv.org/pdf/1512.03385.pdf
class ResnetBottleneckBlock(Module):
    def __init__(self, F1, F2, F3, identity_map=None, kernel_size=(3,3), stride=(1,1)):
        super().__init__()

        self.C1 = nn.Conv2d(F1, F2, kernel_size=(1,1), stride=(1,1), bias=False)
        self.C1_BN = nn.BatchNorm2d(F2)
        self.C1_ACT = nn.ReLU(inplace=True)

        self.C2 = nn.Conv2d(F2, F3, kernel_size=kernel_size, stride=stride, bias=False, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.C2_BN = nn.BatchNorm2d(F3)
        self.C2_ACT = nn.ReLU(inplace=True)

        self.C3 = nn.Conv2d(F3, F3, kernel_size=(1,1), stride=(1,1), bias=False)
        self.C3_BN = nn.BatchNorm2d(F3)

        self.identity_map = identity_map

        self.post_block_act = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        x = self.C1(x)
        x = self.C1_BN(x)
        x = self.C1_ACT(x)

        x = self.C2(x)
        x = self.C2_BN(x)
        x = self.C2_ACT(x)

        x = self.C3(x)
        x = self.C3_BN(x)

        if self.identity_map:
            residual = self.identity_map(residual)

        x += residual

        x = self.post_block_act(x)

        return x
