import torch
import torch.nn as nn
import argparse
from .base_semantic import BaseSemantic

class DistMult(BaseSemantic):

	def __init__(self, config: argparse.Namespace, margin = None, epsilon = None):
		super(DistMult, self).__init__(config)

		self.dim = config.dim
		self.margin = margin
		self.epsilon = epsilon

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embed.weight.data)
			nn.init.xavier_uniform_(self.rel_embed.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embed.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embed.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

	def _calc(self, h, t, r, mode):
		"""
        _calc defines the main methods to calculate the score

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :param mode: char type, 'normal' or 'head_batch'
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h * (r * t)
		else:
			score = (h * r) * t
		score = torch.sum(score, -1).flatten()
		return score

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
		mode = 'normal'
		h = self.ent_embed(batch_h)
		t = self.ent_embed(batch_t)
		r = self.rel_embed(batch_r)
		score = self._calc(h ,t, r, mode)
		return score

