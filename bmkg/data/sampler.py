import random
from typing import Any, Generator, Union, Tuple, Iterable, Iterator
from bmkg._data import TripleDataBatch
from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[TripleDataBatch, TripleDataBatch]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        pass


class RandomCorruptSampler(Sampler):
    def __init__(self, it: Iterable[TripleDataBatch], ent_size: int, mode='head'):
        if mode not in ["head", "tail"]:
            raise ValueError("Invalid sampling mode!")
        self.it = it
        self.mode = mode
        self.ent_size = ent_size

    def __iter__(self):
        def inner():
            for pos in self.it:
                if self.mode == "head":
                    h = np.random.randint(0, self.ent_size, pos.h.shape, pos.h.dtype)
                    neg = TripleDataBatch(h, pos.r, pos.t)
                elif self.mode == "tail":
                    t = np.random.randint(0, self.ent_size, pos.h.shape, pos.h.dtype)
                    neg = TripleDataBatch(pos.h, pos.r, t)
                else:
                    # Unreachable
                    raise ValueError("Invalid sampling mode!")
                yield pos, neg
            return None

        return inner()

    def __len__(self):
        return len(self.it)


class RandomChoiceSampler(Sampler):
    def __init__(self, samplers: list[Sampler]):
        self.samplers = samplers

    def __iter__(self):
        def inner():
            iters = [iter(i) for i in self.samplers]
            while True:
                s = random.choice(iters)
                x = next(s)
                yield x

        return inner()

    def __len__(self):
        return len(self.samplers[0])
