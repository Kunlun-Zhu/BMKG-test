import numpy
from .data import TripleDataset, TripleDataModule
from argparse import Namespace

class TripleDataBatch:
    h: numpy.array
    r: numpy.array
    t: numpy.array
    def __init__(self, h: numpy.array, r: numpy.array, t: numpy.array) -> None: ...

class _TripleDataModule:
    def __init__(self, module: TripleDataModule, config: Namespace) -> None: ...
    ...
