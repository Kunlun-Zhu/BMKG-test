import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.bigmodel import CoKE, CoKE_Roberta, CoKE_BMT, KEPLER
from data import KBCDataset, DescDataset
from config import init_coke_net_config, init_train_config, init_kepler_train_config
from trainer_kepler import Trainer
import math
import logging
import argparse
from config.args import ArgumentGroup
from model_center.dataset import DistributedDataLoader


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

task_g = ArgumentGroup(parser, "task", "which task to run.")
task_g.add_arg("do_train", bool, True, "Train")
task_g.add_arg("do_val", bool, True, "Validation")
task_g.add_arg("do_test", bool, False, "Test")

model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
# model_g.add_arg("model_name", str, "coke_roberta", "Model name")
model_g.add_arg("hidden_size", int, 256, "CoKE model config: hidden size, default 256")
model_g.add_arg("num_hidden_layers", int, 12, "CoKE model config: num_hidden_layers, default 12")
model_g.add_arg("num_attention_heads", int, 4, "CoKE model config: num_attention_heads, default 4")
model_g.add_arg("num_relations", int, 1345, "CoKE model config: vocab_size")
model_g.add_arg("max_position_embeddings", int, 40, "CoKE model config: max_position_embeddings")
model_g.add_arg("dropout", float, 0.1, "CoKE model config: dropout, default 0.1")
model_g.add_arg("hidden_dropout", float, 0.1, "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("attention_dropout", float, 0.1,
                "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("initializer_range", int, 0.02, "CoKE model config: initializer_range")
model_g.add_arg("intermediate_size", int, 512, "CoKE model config: intermediate_size, default 512")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("node", int, 1, "Node nums.")
train_g.add_arg("warmup_epoch", int, 40, "Number of epoches for training.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
# train_g.add_arg("weight_decay", float, 0.0001, "Weight decay rate for L2 regularizer.")
train_g.add_arg("ema_decay", float, 0.9999, "EMA decay.")
train_g.add_arg("lr_scheduler", str, "warmup_cosine", "linear_lr, step_lr")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 30, "The steps interval to print loss.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("dataset", str, "FB15K", "dataset name")
data_g.add_arg("vocab_file", str, 'vocab.txt', "Vocab path.")
data_g.add_arg("train_file", str, 'train.txt', "Train data.")
data_g.add_arg("valid_file", str, 'valid.txt', "Valid data.")
data_g.add_arg("test_file", str, 'test.txt', "Test data.")
data_g.add_arg("desc_file", str, 'desc.txt', "Desc data.")
data_g.add_arg("dev_file", str, 'dev.coke.txt', "Dev data for coke.")
data_g.add_arg("true_triple_file", str, 'all.txt', "All triple data for coke.")
data_g.add_arg("sen_candli_file", str, 'sen_candli.txt',
               "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
data_g.add_arg("sen_trivial_file", str, 'trival_sen.txt',
               "trivial sentence file for pathquery evaluation. Only used for path query datasets")

# parser.add_argument("--task_name", default='path', type=str, required=True, help="path or triple.")
parser.add_argument("--data_root", default='/home/wanghuadong/liangshihao/KEPLER-huggingface/data/', type=str, required=True, help="data directory.")
parser.add_argument("--model_root", default='/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base/', type=str, required=True, help="model directory.")
parser.add_argument("--vocab_size", default=16396, type=int, required=True, help="16396 for fb15k, 75169 for pathFB.")
parser.add_argument("--max_seq_len", default=7, type=int, required=True, help="sequence length.")
parser.add_argument("--epoch", default=400, type=int, required=True, help="epoch.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=2048, type=int, required=True, help="batch size.")
parser.add_argument("--checkpoint_num", default=50, type=int, required=True, help="ckpt nums")
parser.add_argument("--save_path", default='./checkpoints/', type=str, required=True, help="save directory.")
parser.add_argument("--gpu_ids", default='0', type=str, required=True, help="gpu ids.")
parser.add_argument("--model_name", default='kepler', type=str, required=True, help="coke or coke_roberta.")
parser.add_argument("--bmtrain", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--num_classes", default=2, type=int, required=True, help="sentence classsification task.")
parser.add_argument("--base_model", default=None, type=str, required=True, help="base model.")
parser.add_argument("--pooler_dropout", default=0.2, type=float, required=True, help="base model.")
parser.add_argument("--gamma", default=0.2, type=float, required=True, help="gamma.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="gamma.")
parser.add_argument("--nrelation", default=10, type=int, required=True, help="base model.")
parser.add_argument("--ke_model", default='TransE', type=str, required=True, help="ke model.")
parser.add_argument("--padding_idx", default=100, type=int, required=True, help="padding idx in vocab.")
parser.add_argument("--learning_rate", default=0.0001, type=float, required=True, help="lr.")
args = parser.parse_args()


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    print(args.use_cuda)
    if args.bmtrain:
        import bmtrain as bmt
        bmt.init_distributed(seed=0)

    if "kepler" in args.model_name:
        # ------------
        # data
        # ------------
        args.train_data_path = os.path.join(args.data_root, args.train_file)
        args.valid_data_path = os.path.join(args.data_root, args.valid_file)
        args.vocab_path = os.path.join(args.data_root, args.vocab_file)
        args.desc_path = os.path.join(args.data_root, args.desc_file)
        train_dataset = DescDataset(vocab_path=args.vocab_path,
                                    desc_path=args.desc_path,
                                    data_path=args.train_data_path,
                                    max_seq_len=args.max_seq_len,
                                    vocab_size=args.vocab_size)

        val_dataset = DescDataset(vocab_path=args.vocab_path,
                                  desc_path=args.desc_path,
                                  data_path=args.valid_data_path,
                                  max_seq_len=args.max_seq_len,
                                  vocab_size=args.vocab_size)
        if args.bmtrain:
            train_loader = DistributedDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DistributedDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        args.padding_id = train_dataset.pad_id
        train_config = init_kepler_train_config(args, logger, print_config=True)

        # ------------
        # model
        # ------------
        model_config = init_kepler_train_config(args, logger, print_config=True)
        model = KEPLER(config=args)
        state_dict = torch.load(args.model_root+'pytorch_model.bin')
        model.lm_head.dense.weight = nn.Parameter(state_dict['lm_head.dense.weight'])
        model.lm_head.dense.bias = nn.Parameter(state_dict['lm_head.dense.bias'])
        model.lm_head.layer_norm.weight = nn.Parameter(state_dict['lm_head.layer_norm.weight'])
        model.lm_head.layer_norm.bias = nn.Parameter(state_dict['lm_head.layer_norm.bias'])
        model.lm_head.bias = nn.Parameter(state_dict['lm_head.bias'])
        model.lm_head.decoder.weight = nn.Parameter(state_dict['lm_head.decoder.weight'])
        model.lm_head.decoder.bias = nn.Parameter(state_dict['lm_head.decoder.bias'])
        # for name, parms in model.named_parameters():
        #     if 'lm_head.dense' in name:
        #         print('-->name:', name)
        #         print('-->param:', parms)

        if args.bmtrain:
            optimizer = bmt.optim.AdamOptimizer(model.parameters(), lr=args.learning_rate,
                                                weight_decay=args.weight_decay)
            scheduler = bmt.lr_scheduler.Noam(
                optimizer,
                start_lr=args.learning_rate,
                warmup_iter=args.warmup_epoch,
                end_iter=args.epoch,
                num_iter=args.warmup_epoch / 2
            )
            loss_function = bmt.loss.FusedCrossEntropy(ignore_index=train_dataset.pad_id)
            bmt.synchronize()
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # warm_up_with_cosine_lr
            t = args.warmup_epoch  # warmup
            T = args.epoch
            n_t = 0.5
            lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t)))
            if args.lr_scheduler == "warmup_cosine":
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            elif args.lr_scheduler == "linear_lr":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                              total_iters=args.warmup_epoch)
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        trainer = Trainer(model, train_config)

        trainer.load_train_data_loader(train_loader)
        trainer.load_val_data_loader(val_loader)

        trainer.set_optimizer(optimizer)
        trainer.set_lr_scheduler(scheduler)

        total_acc, total_loss = trainer.train()


if __name__ == '__main__':
    main()
