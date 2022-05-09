import abc
import argparse
import enum
import logging
from tkinter import E
from typing import Tuple, ClassVar, Type, Union, Optional

import numpy
import torch.optim

import bmkg.data
from ..model import BMKGModel
from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F

from ...data import TripleDataModule, TripleDataBatchGPU, DataModule
from IPython import embed
import bmtrain as bmt
import math
import torch
from ..bmtlayers import Embedding


class TransX(BMKGModel, ABC):
    data_module: TripleDataModule
    score_name = "hits10"

    def __init__(self, config: argparse.Namespace):
        super(TransX, self).__init__(config)
        self.ranks: list[torch.Tensor] = []
        self.raw_ranks: list[torch.Tensor] = []
        #self.ent_embed = nn.Embedding(config.ent_size, config.dim, max_norm=1)
        #self.rel_embed = nn.Embedding(config.rel_size, config.dim, max_norm=1)
        self.ent_embed = Embedding(config.ent_size, config.dim, max_norm=1)
        self.rel_embed = Embedding(config.rel_size, config.dim, max_norm=1)
        #nn.init.xavier_uniform_(self.ent_embed.weight.data)
        #nn.init.xavier_uniform_(self.rel_embed.weight.data)
        self.gamma = torch.Tensor([config.gamma]).cuda()
        self.p_norm = config.p_norm
        with torch.no_grad():
            ###todo:bmtrain norm
            self.rel_embed.weight /= torch.norm(self.rel_embed.weight.gather(), p=self.p_norm, dim=-1)[:, None]

    @abstractmethod
    def scoring_function(self, heads, rels, tails, *args):
        """
        scoring_function defines a scoring function for a TransX-like model.

        :param heads: torch.Tensor() shaped (batch_size) or (batch_size, neg_sample_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size) or (batch_size, neg_sample_size), containing the id for the tail entity.
        :param args: Additional arguments given by dataset.

        When training, both heads and tails are (batch_size).
        When testing, one of heads and tails are (batch_size, neg_sample_size).
        You should use boardcasting operation to calculate score.

        :return: torch.Tensor() shaped (batch_size) or (batch_sizem neg_sample_size), depending on whether training or testing.
        The individual score for each
        """

    def train_step(self, batch):
        pos, neg = self.forward(*batch)
        neg: torch.Tensor = neg.mean(-1)
        # TODO: regularization
        if self.config.loss_method == 'dgl':
            # dgl-ke style loss
            score = F.logsigmoid(self.gamma - pos) + F.logsigmoid(-neg)
            loss = -score.mean()
        elif self.config.loss_method == 'openke':
            # openke style loss
            loss = (torch.max(pos - neg, -self.gamma)).mean() + self.gamma
        elif self.config.loss_method == 'relu':
            loss = F.relu(pos - neg + self.gamma).mean()
        else:
            raise ValueError("Invalid loss function")
        self.log("train/loss", loss)
        self.log("train/pos_score", pos.mean())
        self.log("train/neg_score", neg.mean())
        return loss

    def on_valid_start(self) -> None:
        self.ranks = []
        self.raw_ranks = []

    def on_valid_end(self) -> None:
        ranks = torch.cat(self.ranks)
        self.log('val/filt/MRR', torch.sum(1.0 / ranks) / ranks.shape[0])
        self.log('val/filt/MR', torch.sum(ranks) / ranks.shape[0])
        self.log('val/filt/hit1', torch.sum(ranks <= 1) / ranks.shape[0])
        self.log('val/filt/hit3', torch.sum(ranks <= 3) / ranks.shape[0])
        self.log('val/filt/hit10', torch.sum(ranks <= 10) / ranks.shape[0])
        raw_ranks = torch.cat(self.raw_ranks)
        self.log('val/raw/MRR', torch.sum(1.0 / raw_ranks) / raw_ranks.shape[0])
        self.log('val/raw/MR', torch.sum(raw_ranks) / raw_ranks.shape[0])
        self.log('val/raw/hit1', torch.sum(raw_ranks <= 1) / raw_ranks.shape[0])
        self.log('val/raw/hit3', torch.sum(raw_ranks <= 3) / raw_ranks.shape[0])
        self.log('val/raw/hit10', torch.sum(raw_ranks <= 10) / raw_ranks.shape[0])
        if ranks.shape != raw_ranks.shape:
            print("Shape mismatch!")
        if torch.any(ranks > raw_ranks):
            print(ranks > raw_ranks)
        # push hit@10 to scores for early_stopping
        self.scores.append(torch.sum(ranks <= 10) / ranks.shape[0])

    def valid_step(self, batch):
        pos: TripleDataBatchGPU = batch
        index = torch.arange(pos.r.shape[0], device=pos.r.device)
        pos_score: torch.Tensor = self.forward(pos)
        # corrupt head
        neg_data = TripleDataBatchGPU(
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)),
            pos.r.view(-1, 1),
            pos.t.view(-1, 1))
        neg_score: torch.Tensor = self.forward(neg_data)
        neg_score[index, pos.h.long()] = float('inf')  # filter positive head itself.
        raw_rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1) + 1
        for idx, (h, r) in enumerate(zip(pos.cpu.h, pos.cpu.r)):
            head_map = self.data_module.get_head_map()
            neg_score[idx][head_map[(h, r)]] = float('inf')
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1) + 1
        self.raw_ranks.append(raw_rank)
        self.ranks.append(rank)

        # corrupt tail
        neg_data = TripleDataBatchGPU(
            pos.h.view(-1, 1), pos.r.view(-1, 1),
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)))
        neg_score: torch.Tensor = self.forward(neg_data)
        neg_score[index, pos.h.long()] = float('inf')  # filter positive tail itself.
        raw_rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1) + 1
        for idx, (t, r) in enumerate(zip(pos.cpu.t, pos.cpu.r)):
            tail_map = self.data_module.get_tail_map()
            neg_score[idx][tail_map[(t, r)]] = float('inf')
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1) + 1
        self.raw_ranks.append(raw_rank)
        self.ranks.append(rank)

    def on_test_start(self) -> None:
        self.ranks = []
        self.raw_ranks = []

    def on_test_end(self) -> None:
        ranks = torch.cat(self.ranks)
        self.log('test/MRR', torch.sum(1.0 / ranks) / ranks.shape[0])
        self.log('test/MR', torch.sum(ranks) / ranks.shape[0])
        self.log('test/hit1', torch.sum(ranks <= 1) / ranks.shape[0])
        self.log('test/hit3', torch.sum(ranks <= 3) / ranks.shape[0])
        self.log('test/hit10', torch.sum(ranks <= 10) / ranks.shape[0])
        raw_ranks = torch.cat(self.raw_ranks)
        self.log('test/raw/MRR', torch.sum(1.0 / raw_ranks) / raw_ranks.shape[0])
        self.log('test/raw/MR', torch.sum(raw_ranks) / raw_ranks.shape[0])
        self.log('test/raw/hit1', torch.sum(raw_ranks <= 1) / raw_ranks.shape[0])
        self.log('test/raw/hit3', torch.sum(raw_ranks <= 3) / raw_ranks.shape[0])
        self.log('test/raw/hit10', torch.sum(raw_ranks <= 10) / raw_ranks.shape[0])

    def test_step(self, batch):
        self.valid_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.config.optim == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.config.optim == "Adam":
            return bmt.optim.AdamOptimizer(self.parameters(), lr=self.lr, weight_decay=0, scale=2**20)
        elif self.config.optim == "Bmtrain":
            optimizer = bmt.optim.AdamOffloadOptimizer(self.parameters(), weight_decay=1e-2, scale=2**20)

    def forward(self, pos: TripleDataBatchGPU, neg: Optional[TripleDataBatchGPU] = None) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # TODO: Data
        if neg is not None:
            pos_score = self.scoring_function(pos.h, pos.r, pos.t)
            neg_score = self.scoring_function(neg.h, neg.r, neg.t)
            return pos_score, neg_score
        else:
            pos_score = self.scoring_function(pos.h, pos.r, pos.t)
            return pos_score

    def on_epoch_start(self):
        with torch.no_grad():
            self.ent_embed.weight /= torch.norm(self.ent_embed.weight, p=self.p_norm, dim=-1)[:, None]

    @staticmethod
    def load_data() -> Type[DataModule]:
        return bmkg.data.TripleDataModule

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = super().add_args(parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--dim", type=int, default=128, help="The embedding dimension for relations and entities")
        parser.add_argument("--gamma", type=float, default=15.0, help="The gamma for max-margin loss")
        parser.add_argument("--p_norm", type=int, default=2, help="The order of the Norm")
        parser.add_argument("--optim", choices=["SGD", "Adam", "Bmtrain"], default="SGD", help="The optimizer to use")
        parser.add_argument("--norm-ord", default=2, help="Ord for norm in scoring function")
        parser.add_argument("--loss_method", default='relu', choices=['dgl', 'openke', 'relu'],
                            help="Which loss function to use")
        return parser
