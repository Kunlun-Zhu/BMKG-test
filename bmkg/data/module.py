import argparse
import pathlib
from abc import ABC
from typing import Iterable, Any
import json

from torch.utils.data import DataLoader

from .data import TripleDataBatchGPU
from .dataset import TripleDataset
from .._data import TripleDataBatch
from .._data import _TripleDataModule

from IPython import embed

class DataModule(ABC):
    """
    DataModule is responsible for constructing train, valid and test DataSets from command line arguments.
    """
    train: Iterable[Any]
    valid: Iterable[Any]
    test: Iterable[Any]

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...


class TripleDataModule(DataModule):
    train: DataLoader
    valid: DataLoader
    test: DataLoader

    train_set: TripleDataset
    valid_set: TripleDataset
    test_set: TripleDataset

    def __init__(self, config: argparse.Namespace):
        self.config = config
        if len(config.data_files) not in [1, 3]:
            raise ValueError("There should be 1 or 3 data files!")
        path = pathlib.Path(config.data_path)
        if len(config.data_files) == 1:
            self.train_set = TripleDataset(path / config.data_files[0], batch_size=config.train_batch_size)
            self.train = DataLoader(
                self.train_set,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn(negative_sample=True)
            )
        else:
            self.train_set = TripleDataset(path / config.data_files[0], batch_size=config.train_batch_size)
            self.valid_set = TripleDataset(path / config.data_files[1], batch_size=config.test_batch_size)
            self.test_set = TripleDataset(path / config.data_files[2], batch_size=config.test_batch_size)
            self.train = DataLoader(
                self.train_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn(negative_sample=True)
            )
            self.valid = DataLoader(
                self.valid_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn()
            )
            self.test = DataLoader(
                self.test_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn()
            )

        with open(path / "config.json") as f:
            data_conf = json.load(f)
            config.ent_size = data_conf['ent_size']
            config.rel_size = data_conf['rel_size']

        self._module = _TripleDataModule(self, config)
        # TODO: Initialize batch
        # TODO: 回忆一下 『Initialize batch』 是要干啥来着

    def collate_fn(self, negative_sample=False):
        def _collate_fn(pos: TripleDataBatch):
            if not negative_sample:
                data = TripleDataBatchGPU(pos)
                return data
            else:
                neg = self._module.neg_sample(pos)
                pos = TripleDataBatchGPU(pos)
                neg = TripleDataBatchGPU(neg)
                return pos, neg

        return _collate_fn

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--data_path', type=str, help="Data path")
        parser.add_argument('--train_batch_size', type=int, default=200, help="Train batch size")
        parser.add_argument('--test_batch_size', type=int, default=50, help="Test batch size")
        parser.add_argument('--train_neg_sample', type=int, default=64, help="Number of negative samples while training")
        parser.add_argument('--data_files', type=str, nargs='+', help="Data filename, e.g. train.npy. If 1 file were"
                                                                      "given, it is treated as the training file, or "
                                                                      "evaluation file during evaluation. If 3 files "
                                                                      "were given, they are treated as training, "
                                                                      "validating and testing data respectively.")
        return parser
