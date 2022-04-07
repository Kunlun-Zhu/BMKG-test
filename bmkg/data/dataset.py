import math
from abc import ABC
from typing import Any, Generator

import numpy as np
import random

from .._data import TripleDataBatch


class TripleDataset:
    """
    Dataset is responsible for reading given data file and yield DataBatch.

    TripleDataset yields TripleDataBatch from a specific range of a given .npy file.
    """

    def __init__(self, filename: str, start: int = 0, end: int = -1, batch_size: int = 20, shuffle: bool = False):
        super(TripleDataset).__init__()
        assert start >= 0
        self.data = np.load(filename)
        if end == -1:
            end = self.data.shape[0]
        assert start < end
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Generator[TripleDataBatch, Any, None]:
        iter_start = self.start
        iter_end = self.end

        def iterator():
            starts = list(range(iter_start, iter_end, self.batch_size))
            if self.shuffle:
                random.shuffle(starts)
            while True:
                for cur in starts:
                    batch = self.data[cur: cur + self.batch_size]
                    data = TripleDataBatch(batch[:, 0], batch[:, 1], batch[:, 2])
                    yield data

        return iterator()

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)
