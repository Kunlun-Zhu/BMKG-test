import torch
import torch.nn as nn
import argparse
from .base_semantic import BaseSemantic

class ComplEx(BaseSemantic):
    def __init__(self, config: argparse.Namespace):
        super(ComplEx, self).__init__(config)

        self.dim = config.dim
        self.ent_re_embeddings = nn.Embedding(config.ent_size, self.dim)
        self.ent_im_embeddings = nn.Embedding(config.ent_size, self.dim)
        self.rel_re_embeddings = nn.Embedding(config.rel_size, self.dim)
        self.rel_im_embeddings = nn.Embedding(config.rel_size, self.dim)

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, h_re, h_im, t_re, t_im, r_re, r_im):
        """
        _calc defines the main methods to calculate the score
        :param h_re: torch.Tensor() shaped (dim), containing the trasnform of head embeddings
        :param h_im: torch.Tensor() shaped (dim), containing the trasnform of head embeddings
        :param t_re: torch.Tensor() shaped (dim), containing the trasnform of tail embeddings
        :param t_im: torch.Tensor() shaped (dim), containing the trasnform of tail embeddings
        :param r_re: torch.Tensor() shaped (dim), containing the trasnform of relation embeddings
        :param r_im: torch.Tensor() shaped (dim), containing the trasnform of relation embeddings
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

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
        h_re = self.ent_re_embeddings(batch_h)
        h_im = self.ent_im_embeddings(batch_h)
        t_re = self.ent_re_embeddings(batch_t)
        t_im = self.ent_im_embeddings(batch_t)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        score = self._calc(h_re, h_im, t_re, t_im, r_re, r_im)
        return score

