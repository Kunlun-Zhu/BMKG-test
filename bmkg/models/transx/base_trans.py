import argparse

import torch.optim

from ..model import BMKGModel
from abc import ABC, abstractmethod
from torch import nn


class BaseTrans(BMKGModel, ABC):
    def __init__(self, config: argparse.Namespace):
        super(TransX, self).__init__()
        self.ent_embed = nn.Embedding(config.ent_size, config.emb_dim)
        self.rel_embed = nn.Embedding(config.rel_size, config.rel_dim)
        self.gamma = config.gamma

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

    def train_step(self, data):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, pos, neg):
        pos_score = self.scoring_funcion(pos)
        neg_score = self.scoring_funcion(neg)
        # we want minimal loss
        loss = self.gamma - pos_score + neg_score
        return loss

    def load_data(self):
        """
        Loads TripleDataset or it's subclasses.
        :return: bmkg.data.TripleDataset
        """
        raise NotImplementedError
