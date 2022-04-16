import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .transx import TransX

class TransR(TransX):

	def __init__(self,config: argparse.Namespace, p_norm = 1, norm_flag = True, rand_init = False, margin = None):
		super(TransR, self).__init__(config)
		
		self.dim_e = config.dim
		self.dim_r = config.dim
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.rand_init = rand_init
		self.ent_size = config.ent_size

		nn.init.xavier_uniform_(self.ent_embed.weight.data)
		nn.init.xavier_uniform_(self.rel_embed.weight.data)

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

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _calc(self, h, t, r, mode):
		"""
		_calc defines the main methods to calculate the score

		:param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
		:param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
		:param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
		:return: torch.Tensor() shaped (batch_size). The individual score for each
		"""
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def _transfer(self, e, r_transfer):
		r_transfer = r_transfer.view(-1, self.dim_e, self.dim_r)
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
			e = torch.matmul(e, r_transfer).permute(1, 0, 2)
		else:
			e = e.view(-1, 1, self.dim_e)
			e = torch.matmul(e, r_transfer)
		return e.view(-1, self.dim_r)
	
	def _calc_2(self, h, t, r, mode):
		"""
		score calculation for valid & test
		_calc defines the main methods to calculate the score

		:param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
		:param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
		:param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
		:return: torch.Tensor() shaped (batch_size). The individual score for each
		"""
		
		score = (h + r) - t
		
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	
	def _transfer_2(self, e, r_transfer):
		#transfer method for valid&test
		r_transfer = r_transfer.view(-1, 1, self.dim_e, self.dim_r)

		e = e.view(1, -1, 1, self.dim_e)

		e = torch.matmul(e, r_transfer)
		
		print (e.shape)

		e = e.squeeze(dim=2)

		print (e.shape)

		return e

	#scoring function added
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
		r_transfer = self.transfer_matrix(batch_r)
		h = self._transfer(h, r_transfer)
		t = self._transfer(t, r_transfer)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def scoring_function_2(self, h, r, t):

		"""
		score function for valid_step & test step
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
		r_transfer = self.transfer_matrix(batch_r)
		h = self._transfer_2(h, r_transfer)
		t = self._transfer_2(t, r_transfer)
		score = self._calc_2(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score


