import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .transx import TransX
from ..bmtlayers import Embedding
import bmtrain as bmt

class TransR(TransX):

    def __init__(self, config: argparse.Namespace, norm_flag=True, rand_init=False):
        super(TransR, self).__init__(config)

        self.dim_e = config.dim
        self.dim_r = config.dim
        self.norm_flag = norm_flag
        self.rand_init = rand_init
        self.ent_size = config.ent_size

        self.transfer_matrix = Embedding(config.rel_size, self.dim_e * self.dim_r)
        
        if not self.rand_init:
            identity = torch.eye(self.dim_e, self.dim_r)
            self.transfer_matrix.weight.data = identity \
                .expand(config.rel_size, self.dim_e, self.dim_r) \
                .view(config.rel_size, self.dim_e * self.dim_r)
        else:
            nn.init.xavier_uniform_(self.transfer_matrix.weight.data)


    def scoring_function(self, heads, rels, tails, *_):
        """
        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """
        # (batch_size, 1, dim_e) or (batch_size, dim_e)
        h: torch.Tensor = self.ent_embed(heads)
        # (batch_size, 1, dim_e) or (batch_size, dim_e)
        r: torch.Tensor = self.rel_embed(rels)
        # (1, ent_size, dim_e) or (batch_size, dim_e)
        t: torch.Tensor = self.ent_embed(tails)
        # (batch_size, 1, dim_e * dim_e) or (batch_size, dim_e * dim_e)
        prj: torch.Tensor = self.transfer_matrix(rels)
        # (batch_size, 1, dim_e, dim_e) or (batch, dim_e, dim_e)
        prj = prj.view(*prj.size()[:-1], self.dim_e, self.dim_r)

        # (batch_size, 1, 1, dim_e) or (batch_size, 1, dim_e)
        h = torch.unsqueeze(h, -2)
        # (1, ent_size, 1, dim_e) or (batch_size, 1, dim_e)
        t = torch.unsqueeze(t, -2)

        # (batch_size, 1, 1, dim_e) or (batch_size, 1, dim_e)
        h = torch.matmul(h, prj)
        # (1, ent_size, 1, dim_e) or (batch_size, 1, dim_e)
        t = torch.matmul(t, prj)

        # (batch_size, 1, dim_e) or (batch_size, dim_e)
        h = torch.squeeze(h, -2)
        # (1, ent_size, dim_e) or (batch_size, dim_e)
        t = torch.squeeze(t, -2)

        # (batch_size, ent_size, dim_e) or (batch_size, dim_e)
        score = h + r - t
        score = torch.norm(score, p=self.p_norm, dim=-1)
        return score