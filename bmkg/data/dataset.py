import math
from abc import ABC
from typing import Any, Generator

import numpy as np
import random
import torch
from .._data import TripleDataBatch


class TripleDataset(torch.utils.data.IterableDataset):
    """
    Dataset is responsible for reading given data file and yield DataBatch.

    TripleDataset yields TripleDataBatch from a specific range of a given .npy file.
    """

    def __init__(self, filename: str, start: int = 0, end: int = -1, batch_size: int = 20, shuffle: bool = False, loop: bool = True):
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
        self.loop = loop

    def __iter__(self) -> Generator[TripleDataBatch, Any, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        def iterator():
            starts = list(range(iter_start, iter_end, self.batch_size))
            if self.shuffle:
                random.shuffle(starts)
            for cur in starts:
                batch = self.data[cur: cur + self.batch_size]
                data = TripleDataBatch(batch[:, 0], batch[:, 1], batch[:, 2])
                yield data
        return iterator()

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)
