import argparse
import pathlib
from abc import ABC
from typing import Iterable, Any, Union
import json
import numpy
import torch
import bmtrain as bmt
from torch.utils.data import DataLoader
import torch.utils.data as data
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
            
            self.train_sampler = data.distributed.DistributedSampler(self.train_set, shuffle=False, rank=bmt.rank(), num_replicas=bmt.world_size())
            self.valid_sampler = data.distributed.DistributedSampler(self.valid_set, shuffle=False, rank=bmt.rank(), num_replicas=bmt.world_size())
            self.test_sampler = data.distributed.DistributedSampler(self.test_set, shuffle=False, rank=bmt.rank(), num_replicas=bmt.world_size())
            

            self.train = DataLoader(
                self.train_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn(negative_sample=True),
                shuffle=False,
                sampler=self.train_sampler
            )
            self.valid = DataLoader(
                self.valid_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn(),
                shuffle=False,
                sampler=self.valid_sampler
            )
            self.test = DataLoader(
                self.test_set,
                # num_workers=1,
                batch_size=None,  # we don't use torch's auto data batching
                collate_fn=self.collate_fn(),
                shuffle=False,
                sampler=self.test_sampler
            )

        with open(path / "config.json") as f:
            data_conf = json.load(f)
            config.ent_size = data_conf['ent_size']
            config.rel_size = data_conf['rel_size']

        self._module = _TripleDataModule(self, config)
        # TODO: Initialize batch
        # TODO: 回忆一下 『Initialize batch』 是要干啥来着
    
    # def calc_rank(self, mode: Union["head", "tail"], pos_data: TripleDataBatch, pos_score: torch.Tensor, neg_score: torch.Tensor):
    #     return self._module.calc_rank(mode, pos_data.cpu, pos_score.cpu().numpy(), neg_score.cpu().numpy())

    # def gather_ranks(self):
    #     return torch.from_numpy(self._module.gather_ranks()).cuda()
    
    def get_head_map(self):
        return self._module.head_map
    
    def get_tail_map(self):
        return self._module.tail_map

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
        parser.add_argument('--test_batch_size', type=int, default=100, help="Test batch size")
        parser.add_argument('--train_neg_sample', type=int, default=64, help="Number of negative samples while training")
        parser.add_argument('--data_files', type=str, nargs='+', help="Data filename, e.g. train.npy. If 1 file were"
                                                                      "given, it is treated as the training file, or "
                                                                      "evaluation file during evaluation. If 3 files "
                                                                      "were given, they are treated as training, "
                                                                      "validating and testing data respectively.")
        return parser
