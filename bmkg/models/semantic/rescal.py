import torch
import torch.nn as nn
import argparse
from .semantic import BaseSemantic

class RESCAL(BaseSemantic):

	def __init__(self, config: argparse.Namespace, dim = 100):
		super(RESCAL, self).__init__(config)

		self.dim = dim
		self.ent_embeddings = nn.Embedding(config.ent_size, self.dim)
		self.rel_matrices = nn.Embedding(config.rel_size, self.dim * self.dim)

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)
	
	def _calc(self, h, t, r):
		"""
        _calc defines the main methods to calculate the score

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :param mode: char type, 'normal' or 'head_batch'
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """
		t = t.view(-1, self.dim, 1)
		r = r.view(-1, self.dim, self.dim)
		tr = torch.matmul(r, t)
		tr = tr.view(-1, self.dim)
		return -torch.sum(h * tr, -1)

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
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		score = self._calc(h ,t, r)
		return score

