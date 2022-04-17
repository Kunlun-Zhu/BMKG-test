import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .transx import TransX

class TransR(TransX):

	def __init__(self,config: argparse.Namespace, norm_flag = True, rand_init = False, margin = None):
		super(TransR, self).__init__(config)
		
		self.dim_e = config.dim
		self.dim_r = config.dim
		self.norm_flag = norm_flag
		self.rand_init = rand_init
		self.ent_size = config.ent_size

		self.transfer_matrix = nn.Embedding(config.rel_size, self.dim_e * self.dim_r)
		if not self.rand_init:
			identity = torch.zeros(self.dim_e, self.dim_r)
			for i in range(min(self.dim_e, self.dim_r)):
				identity[i][i] = 1
			identity = identity.view(self.dim_r * self.dim_e)
			for i in range(config.rel_size):
				self.transfer_matrix.weight.data[i] = identity
		else:
			nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

		if margin is not None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def scoring_function(self, heads, rels, tails, *_):
		"""
		:param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
		:param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
		:param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
		:return: torch.Tensor() shaped (batch_size). The individual score for each
		"""

		h: torch.Tensor = self.ent_embed(heads)
		r: torch.Tensor = self.rel_embed(rels)
		t: torch.Tensor = self.ent_embed(tails)
		prj: torch.Tensor = self.transfer_matrix(rels)
		prj = prj.view(*prj.size()[:-1], self.dim_e, self.dim_r)

		h = torch.unsqueeze(h, -2)
		t = torch.unsqueeze(t, -2)

		h = torch.matmul(h, prj)
		t = torch.matmul(t, prj)


		h = torch.squeeze(h, -2)
		t = torch.squeeze(t, -2)

		score = h + r - t
		score = torch.norm(score, p=self.p_norm, dim=-1)
		return score
