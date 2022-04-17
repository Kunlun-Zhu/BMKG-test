import abc
import argparse
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

from ...data import TripleDataBatch, TripleDataBatchGPU, DataModule
from IPython import embed



class TransX(BMKGModel, ABC):
    def __init__(self, config: argparse.Namespace):
        super(TransX, self).__init__(config)
        self.ranks: list[torch.LongTensor] = []
        self.ent_embed = nn.Embedding(config.ent_size, config.dim, max_norm=1)
        self.rel_embed = nn.Embedding(config.rel_size, config.dim, max_norm=1)
        nn.init.xavier_uniform_(self.ent_embed.weight.data)
        nn.init.xavier_uniform_(self.rel_embed.weight.data)
        self.gamma = torch.Tensor([config.gamma]).cuda()
        self.p_norm = config.p_norm
        with torch.no_grad():
            self.rel_embed.weight /= torch.norm(self.rel_embed.weight, p=self.p_norm, dim=-1)[:, None]

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

    def on_valid_end(self) -> None:
        ranks = torch.cat(self.ranks)
        self.log('val/MRR', torch.sum(1.0 / ranks) / ranks.shape[0])
        self.log('val/MR', torch.sum(ranks) / ranks.shape[0])
        self.log('val/hit1', torch.sum(ranks <= 1) / ranks.shape[0])
        self.log('val/hit3', torch.sum(ranks <= 3) / ranks.shape[0])
        self.log('val/hit10', torch.sum(ranks <= 10) / ranks.shape[0])

    def valid_step(self, batch):
        pos = batch
        pos_score: torch.Tensor = self.forward(pos)
        # corrupt head
        neg_data = TripleDataBatchGPU(
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)), pos.r.view(-1, 1),
            pos.t.view(-1, 1))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1)
        self.ranks.append(rank)
        # corrupt tail
        neg_data = TripleDataBatchGPU(
            pos.h.view(-1, 1), pos.r.view(-1, 1),
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1)
        self.ranks.append(rank)

    def on_test_start(self) -> None:
        self.ranks = []

    def on_test_end(self) -> None:
        ranks = torch.cat(self.ranks)
        self.log('test/MRR', torch.sum(1.0 / ranks) / ranks.shape[0])
        self.log('test/MR', torch.sum(ranks) / ranks.shape[0])
        self.log('test/hit1', torch.sum(ranks <= 1) / ranks.shape[0])
        self.log('test/hit3', torch.sum(ranks <= 3) / ranks.shape[0])
        self.log('test/hit10', torch.sum(ranks <= 10) / ranks.shape[0])

    def test_step(self, batch):
        pos = batch
        pos_score: torch.Tensor = self.forward(pos)
        # corrupt head
        neg_data = TripleDataBatchGPU(
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)), pos.r.view(-1, 1),
            pos.t.view(-1, 1))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1)
        self.ranks.append(rank)
        # corrupt tail
        neg_data = TripleDataBatchGPU(
            pos.h.view(-1, 1), pos.r.view(-1, 1),
            torch.arange(0, self.config.ent_size, dtype=torch.int32, device=pos.h.device).view((1, -1)))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.view(-1, 1), dim=1)
        self.ranks.append(rank)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.config.optim == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.config.optim == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

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
        parser.add_argument("--optim", choices=["SGD", "Adam"], default="SGD", help="The optimizer to use")
        parser.add_argument("--norm-ord", default=2, help="Ord for norm in scoring function")
        parser.add_argument("--loss_method", default='relu', choices=['dgl', 'openke', 'relu'],
                            help="Which loss function to use")
        return parser
