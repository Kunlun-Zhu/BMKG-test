import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from bmkg.data import MaskTripleDataLoader, RandomChoiceMaskSampler, RandomCorruptMaskSampler
# from ..model import BMKGModel
from model_center.layer import Embedding, Linear, LayerNorm, FeedForward, Encoder
import bmtrain as bmt


class CoKE_BMT(bmt.DistributedModule):
    """CoKE: Contextualized Knowledge Graph Embedding."""

    # TODO: soft label, attn_mask scale, activation function.
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.max_seq_len = config['max_seq_len']
        self.emb_size = config['hidden_size']
        self.n_layer = config['num_hidden_layers']
        self.n_head = config['num_attention_heads']
        self.voc_size = config['vocab_size']
        self.n_relation = config['num_relations']
        self.max_position_seq_len = config['max_position_embeddings']
        self.dropout = config['dropout']
        self.attention_dropout = config['attention_dropout']
        self.intermediate_size = config['intermediate_size']
        self.weight_sharing = config['weight_sharing']
        self.initializer_range = config['initializer_range']

        self.word_embedding = Embedding(vocab_size=self.voc_size, embedding_size=self.emb_size,
                                        init_mean=0.0, init_std=self.initializer_range)
        self.position_embedding = Embedding(vocab_size=self.max_position_seq_len, embedding_size=self.emb_size,
                                            init_mean=0.0, init_std=self.initializer_range)
        self.layer_norm = LayerNorm(dim_norm=self.emb_size, eps=1e-12)

        self.encoder = Encoder(
            num_layers=self.n_layer,
            dim_model=self.emb_size,
            dim_ff=self.intermediate_size,
            num_heads=self.n_head,
            dim_head=self.emb_size // self.n_head,
            norm_eps=1e-12,
            dropout_p=self.dropout
        )
        self.ffn = FeedForward(
            dim_in=self.emb_size,
            dim_ff=self.emb_size,
            dim_out=self.voc_size,
            dropout_p=self.dropout
        )
        # self.init_bmt_parameters()

    def init_bmt_parameters(self):
        bmt.init_parameters(self.word_embedding)
        bmt.init_parameters(self.position_embedding)
        bmt.init_parameters(self.layer_norm)
        bmt.init_parameters(self.encoder)
        bmt.init_parameters(self.ffn)

    def load_pretrained_embedding(self, path):
        # self.load_state_dict()
        state_dict = torch.load(os.path.join(path))
        for key, value in state_dict.items():
            if key == 'ent_embeddings.weight':
                entity_embed = value
            if key == 'rel_embeddings.weight':
                rel_embed = value
        coke_embed = torch.cat((entity_embed, rel_embed), dim=0)
        self.embedding_layer["word_embedding"].weight = nn.Parameter(coke_embed)
        self.linear2.weight = self.embedding_layer["word_embedding"].weight

    def forward(self, input_map):
        src_ids = input_map['src_ids'].squeeze()
        position_ids = input_map['position_ids'].squeeze()
        mask_pos = input_map['mask_pos'].squeeze()
        input_mask = input_map['input_mask'].squeeze(dim=1)

        emb_out = self.word_embedding(src_ids) + self.position_embedding(position_ids)
        emb_out = self.layer_norm(emb_out)
        # emb_out = self.embedding_layer["dropout"](emb_out)

        with torch.no_grad():
            self_attn_mask = torch.bmm(input_mask, input_mask.permute(0, 2, 1)) * -10000

        enc_out = self.encoder(emb_out, self_attn_mask)
        # method 1
        # enc_out = enc_out.reshape(shape=[-1, self.emb_size])
        # enc_out = torch.index_select(input=enc_out, dim=0, index=mask_pos)

        # method 2
        enc_out = enc_out[torch.arange(mask_pos.shape[0]), mask_pos, :]
        # method 3
        # enc_out = enc_out.transpose(0, 1)
        # enc_out = enc_out[src_ids == 99]

        logits = self.ffn(enc_out)
        output_map = {
            'logits': logits
        }
        return output_map

    def on_train_start(self):
        head_sampler = RandomCorruptMaskSampler(self.train_data, self.config.ent_size, mode='head')
        tail_sampler = RandomCorruptMaskSampler(self.train_data, self.config.ent_size, mode='tail')
        combined = RandomChoiceMaskSampler([head_sampler, tail_sampler])
        self.train_data = combined

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_step(self, batch):
        input_map = dict()
        input_map['src_ids'] = batch[0]
        input_map['position_ids'] = batch[1]
        input_map['mask_pos'] = batch[2]
        input_map['input_mask'] = batch[3]
        labels = batch[4].squeeze()

        output_map = self.forward(input_map)

        logits = output_map['logits']
        loss = F.cross_entropy(logits, labels, label_smoothing=0.8)
        self.log("train/loss", loss)
        return loss

    @staticmethod
    def load_data():
        return MaskTripleDataLoader

    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--max_seq_len', type=int, default=3, help="Number of tokens of the longest sequence.")
        parser.add_argument('--pad_id', type=int, default=-100, help="<pad> id in vocab")
        parser.add_argument('--hidden_size', type=int, default=256, help="CoKE model config, default 256")
        parser.add_argument('--num_hidden_layers', type=int, default=6, help="CoKE model config, default 6")
        parser.add_argument('--num_attention_heads', type=int, default=4, help="CoKE model config, default 4")
        parser.add_argument('--vocab_size', type=int, default=16396, help="CoKE model config")
        parser.add_argument('--num_relations', type=int, default=0, help="CoKE model config")
        parser.add_argument('--max_position_embeddings', type=int, default=10, help="max position embeddings")
        parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
        parser.add_argument('--attention_dropout', type=float, default=0.1, help="attention dropout")
        parser.add_argument('--intermediate_size', type=float, default=512, help="intermediate size")

        return parser
