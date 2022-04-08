import torch
import torch.nn as nn
import argparse
from .base_semantic import BaseSemantic

class SimplE(BaseSemantic):

    def __init__(self, config: argparse.Namespace):
        super(SimplE, self).__init__(config)

        self.dim = config.dim
        self.rel_inv_embeddings = nn.Embedding(config.rel_size, self.dim)

        nn.init.xavier_uniform_(self.ent_embed.weight.data)
        nn.init.xavier_uniform_(self.rel_embed.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    def _calc_avg(self, h, t, r, r_inv):
        return (torch.sum(h * r * t, -1) + torch.sum(h * r_inv * t, -1))/2

    def _calc_ingr(self, h, r, t):
        return torch.sum(h * r * t, -1)

    def scoring_function(self, h, r, t):
        """

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        
        """
        batch_h = h
        batch_t = t
        batch_r = r
        h = self.ent_embed(batch_h)
        t = self.ent_embed(batch_t)
        r = self.rel_embed(batch_r)
        r_inv = self.rel_inv_embeddings(batch_r)
        score = self._calc_avg(h, t, r, r_inv)
        return score
