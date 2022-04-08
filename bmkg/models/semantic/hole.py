import torch
import torch.nn as nn
from .base_semantic import BaseSemantic
import numpy
import argparse
from numpy import fft

class HolE(BaseSemantic):

	def __init__(self, config: argparse.Namespace, margin = None, epsilon = None):
		super(HolE, self).__init__(config)

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
	
	def _conj(self, tensor):
		zero_shape = (list)(tensor.shape)
		one_shape = (list)(tensor.shape)
		zero_shape[-1] = 1
		one_shape[-1] -= 1
		ze = torch.zeros(size = zero_shape, device = tensor.device)
		on = torch.ones(size = one_shape, device = tensor.device)
		matrix = torch.cat([ze, on], -1)
		matrix = 2 * matrix
		return tensor - matrix * tensor
	
	def _real(self, tensor):
		dimensions = len(tensor.shape)
		return tensor.narrow(dimensions - 1, 0, 1)

	def _imag(self, tensor):
		dimensions = len(tensor.shape)
		return tensor.narrow(dimensions - 1, 1, 1)

	def _mul(self, real_1, imag_1, real_2, imag_2):
		real = real_1 * real_2 - imag_1 * imag_2
		imag = real_1 * imag_2 + imag_1 * real_2
		return torch.cat([real, imag], -1)

	def _ccorr(self, a, b):
		'''
		original ccorr:
		a = self._conj(torch.rfft(a, signal_ndim = 1, onesided = False))
		b = torch.rfft(b, signal_ndim = 1, onesided = False)
		res = self._mul(self._real(a), self._imag(a), self._real(b), self._imag(b))
		res = torch.ifft(res, signal_ndim = 1)
		return self._real(res).flatten(start_dim = -2)
		'''
		a = self._conj(torch.view_as_real(torch.fft.fft(a, dim = 1)))
		
		print(a.shape)
		
		b = torch.view_as_real(torch.fft.fft(b, dim = 1))
		
		print(b.shape)
		
		res = self._mul(self._real(a), self._imag(a), self._real(b), self._imag(b))
		
		print(res.shape)
		
		res = torch.fft.ifft(torch.view_as_complex(res), n=res.shape[1], dim=1)
		
		print(res.shape)
		print(self._real(res).flatten(start_dim = -1).shape)

		return self._real(res).flatten(start_dim = -2)

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
		

		score = self._ccorr(h, t) * r
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
