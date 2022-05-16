import argparse
import torch

from .transx import TransX


class TransE(TransX):
    def __init__(self, config: argparse.Namespace):
        super(TransE, self).__init__(config)

    def scoring_function(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor, *_):
        score = self.ent_embed(heads) + self.rel_embed(rels) - self.ent_embed(tails)
        score = torch.norm(score, p=self.p_norm, dim=-1)
        ###bmtrain norm
        return score

    def scoreing_function_2(self, heads, rels, tails, *_):
        #score function for valid & test remain the same in TransE
        score = self.ent_embed(heads) + self.rel_embed(rels) - self.ent_embed(tails)
        score = torch.norm(score, p=self.p_norm, dim=-1)
        return score
