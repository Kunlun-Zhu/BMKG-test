import abc
import argparse
import logging
from typing import Tuple, ClassVar, Type

import torch.optim

import bmkg.data
from ..model import BMKGModel
from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F

from ...data import DataLoader, TripleDataLoader, RandomCorruptSampler, RandomChoiceSampler


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

    @abstractmethod
    def scoring_function(self, heads, rels, tails, *args):
        """
        scoring_function defines a scoring function for a TransX-like model.

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :param args: Additional arguments given by dataset.
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """

    def train_step(self, batch):
        pos, neg = self.forward(*batch)
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
        neg_data = TripleDataBatch(
            numpy.arange(0, self.config.ent_size, dtype=numpy.int32).reshape((1, -1)), pos.r.reshape(-1, 1), pos.t.reshape(-1, 1))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.reshape(-1, 1), dim=1)
        self.ranks.append(rank)
        # corrupt tail
        neg_data = TripleDataBatch(
            pos.h.reshape(-1, 1), pos.r.reshape(-1, 1), numpy.arange(0, self.config.ent_size, dtype=numpy.int32).reshape((1, -1)))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.reshape(-1, 1), dim=1)
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
        neg_data = TripleDataBatch(
            numpy.arange(0, self.config.ent_size, dtype=numpy.int32).reshape((1, -1)), pos.r.reshape(-1, 1), pos.t.reshape(-1, 1))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.reshape(-1, 1), dim=1)
        self.ranks.append(rank)
        # corrupt tail
        neg_data = TripleDataBatch(
            pos.h.reshape(-1, 1), pos.r.reshape(-1, 1), numpy.arange(0, self.config.ent_size, dtype=numpy.int32).reshape((1, -1)))
        neg_score: torch.Tensor = self.forward(neg_data)
        # todo: remove positive edge when counting.
        rank = torch.sum(neg_score <= pos_score.reshape(-1, 1), dim=1)
        self.ranks.append(rank)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, pos, neg=None) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # TODO: Data
        if neg is not None:
            posh = torch.LongTensor(pos.h).cuda()
            posr = torch.LongTensor(pos.r).cuda()
            post = torch.LongTensor(pos.t).cuda()
            negh = torch.LongTensor(neg.h).cuda()
            negr = torch.LongTensor(neg.r).cuda()
            negt = torch.LongTensor(neg.t).cuda()
            pos_score = self.scoring_function(posh, posr, post)
            neg_score = self.scoring_function(negh, negr, negt)
            return pos_score, neg_score
        else:
            posh = torch.LongTensor(pos.h).cuda()
            posr = torch.LongTensor(pos.r).cuda()
            post = torch.LongTensor(pos.t).cuda()
            pos_score = self.scoring_function(posh, posr, post)
            return pos_score

    def on_epoch_end(self):
        torch.set_grad_enabled(False)
        self.ent_embed.weight /= torch.norm(self.ent_embed.weight, p=self.p_norm, dim=-1)[:, None]
        self.rel_embed.weight /= torch.norm(self.rel_embed.weight, p=self.p_norm, dim=-1)[:, None]
        torch.set_grad_enabled(True)

    def on_train_start(self):
        head_sampler = RandomCorruptSampler(self.train_data, self.config.ent_size, mode='head')
        tail_sampler = RandomCorruptSampler(self.train_data, self.config.ent_size, mode='tail')
        combined = RandomChoiceSampler([head_sampler, tail_sampler])
        self.train_data = combined

    @staticmethod
    def load_data() -> Type[DataLoader]:
        return bmkg.data.TripleDataLoader

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = super().add_args(parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--dim", type=int, default=128, help="The embedding dimension for relations and entities")
        parser.add_argument("--gamma", type=float, default=15.0, help="The gamma for max-margin loss")
        parser.add_argument("--p_norm", type=int, default=2, help="The order of the Norm")
        parser.add_argument("--norm-ord", default=2, help="Ord for norm in scoring function")
        parser.add_argument("--loss_method", default='relu', choices=['dgl', 'openke', 'relu'], help="Which loss function to use")

        return parser
