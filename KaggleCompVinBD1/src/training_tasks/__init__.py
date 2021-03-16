from enum import Enum


class BackpropAggregators(Enum):
    MeanLosses = 0
    IndividualBackprops = 1
