import torch
from torch import nn
from torch.nn import Module

from src.models.components.ResNet.BuildingBlocks import ResnetBuildingBlock, ResnetBottleneckBlock
from src.models.components.ResNet.IdentityMaps import ZeroPad, ProjectionShortcut


class RNC1(Module):
    # The first layer in the ResNet architecture
    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.Conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2))
        self.BN = nn.BatchNorm2d(64)
        self.ACT = nn.ReLU(inplace=True)

        self.Pool = nn.MaxPool2d((3, 3), stride=(2, 2))

    def forward(self, x):

        x = self.Conv(x)
        x = self.BN(x)
        x = self.ACT(x)

        x = self.Pool(x)

        return x


class RNC2Small(Module):
    # The first layer in the ResNet architecture
    def __init__(self, blocks: int = 3):
        super().__init__()

        self.layer = nn.Sequential(*[
            ResnetBuildingBlock(64, (3, 3)),
        ] * blocks)

    def forward(self, x):

        x = self.layer(x)

        return x


class RNC2(Module):
    # The first layer in the ResNet architecture
    def __init__(self, blocks: int = 3):
        super().__init__()

        first_identity_map = ProjectionShortcut(64, 256, stride=(2, 2))
        rest_identity_map = ZeroPad(256, 256)

        self.layer = nn.Sequential(*(
            [ResnetBottleneckBlock(64, 64, 256, identity_map=first_identity_map, stride=(2, 2))] +
            [
                ResnetBottleneckBlock(256, 64, 256, identity_map=rest_identity_map),
            ] * (blocks - 1)
        ))

    def forward(self, x):

        x = self.layer(x)

        return x


class RNC3(Module):
    # The first layer in the ResNet architecture
    def __init__(self, blocks: int = 3):
        super().__init__()

        first_identity_map = ProjectionShortcut(256, 512, stride=(2, 2))
        rest_identity_map = ZeroPad(512, 512)

        self.layer = nn.Sequential(*(
            [ResnetBottleneckBlock(256, 128, 512, identity_map=first_identity_map, stride=(2, 2))] +
            [
                ResnetBottleneckBlock(512, 128, 512, identity_map=rest_identity_map),
            ] * (blocks - 1)
        ))

    def forward(self, x):

        x = self.layer(x)

        return x


class RNC4(Module):
    # The first layer in the ResNet architecture
    def __init__(self, blocks: int = 3):
        super().__init__()

        first_identity_map = ProjectionShortcut(512, 1024, stride=(2, 2))
        rest_identity_map = ZeroPad(1024, 1024)

        self.layer = nn.Sequential(*(
            [ResnetBottleneckBlock(512, 256, 1024, identity_map=first_identity_map, stride=(2, 2))] +
            [
                ResnetBottleneckBlock(1024, 256, 1024, identity_map=rest_identity_map),
            ] * (blocks - 1)
        ))

    def forward(self, x):

        x = self.layer(x)

        return x


class RNC5(Module):
    # The first layer in the ResNet architecture
    def __init__(self, blocks: int = 3):
        super().__init__()

        first_identity_map = ProjectionShortcut(1024, 2048, stride=(2, 2))
        rest_identity_map = ZeroPad(2048, 2048)

        self.layer = nn.Sequential(*(
            [ResnetBottleneckBlock(1024, 512, 2048, identity_map=first_identity_map, stride=(2, 2))] +
            [
                ResnetBottleneckBlock(2048, 512, 2048, identity_map=rest_identity_map),
            ] * (blocks - 1)
        ))

    def forward(self, x):

        x = self.layer(x)

        return x