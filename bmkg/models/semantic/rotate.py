import torch
import torch.autograd as autograd
import torch.nn as nn
import argparse
from .base_semantic import BaseSemantic

class RotatE(BaseSemantic):

	def __init__(self, config: argparse.Namespace, margin = 6.0, epsilon = 2.0):
		super(RotatE, self).__init__(config)

		self.margin = margin
		self.epsilon = epsilon

		self.dim_e = config.dim * 2
		self.dim_r = config.dim

		self.ent_embed = nn.Embedding(config.ent_size, self.dim_e, max_norm=1)

		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

		self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.ent_embed.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

		self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)

		nn.init.uniform_(
			tensor = self.rel_embed.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)

		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False

	def _calc(self, h, t, r, mode):
		"""
        _calc defines the main methods to calculate the score

        :param heads: torch.Tensor() shaped (batch_size), containing the id for the head entity.
        :param rels: torch.Tensor() shaped (batch_size), containing the id for the relation.
        :param tails: torch.Tensor() shaped (batch_size), containing the id for the tail entity.
        :param mode: char type, 'normal' or 'head_batch'
        :return: torch.Tensor() shaped (batch_size). The individual score for each
        """
		pi = self.pi_const

		re_head, im_head = torch.chunk(h, 2, dim=-1)
		re_tail, im_tail = torch.chunk(t, 2, dim=-1)

		phase_relation = r / (self.rel_embedding_range.item() / pi)

		re_relation = torch.cos(phase_relation)
		im_relation = torch.sin(phase_relation)

		re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
		re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
		im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
		im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
		im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
		re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

		if mode == "head_batch":
			re_score = re_relation * re_tail + im_relation * im_tail
			im_score = re_relation * im_tail - im_relation * re_tail
			re_score = re_score - re_head
			im_score = im_score - im_head
		else:
			re_score = re_head * re_relation - im_head * im_relation
			im_score = re_head * im_relation + im_head * re_relation
			re_score = re_score - re_tail
			im_score = im_score - im_tail

		score = torch.stack([re_score, im_score], dim = 0)
		score = score.norm(dim = 0).sum(dim = -1)
		return score.permute(1, 0).flatten()
		
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
		mode = "normal"
		h = self.ent_embed(batch_h)
		t = self.ent_embed(batch_t)
		r = self.rel_embed(batch_r)
		score = self.margin - self._calc(h ,t, r, mode)
		return score