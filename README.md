## Data部分的设计

数据流： 由模型返回DataLoader，DataLoader 根据命令行参数构造DataSet，从DataSet拿到DataBatch后传递给Sampler， Sampler将pos和neg的DataBatch传递给模型。

data.coke_dataset部分用于当前coke调试与评估，后续会删除，将transformer模型部分的data合入dataset、loader、sampler

## For Coke

Requirements:

- torch ==1.11.0

#### to train

```shell
# triple data
# /home/liangshihao/data/fb15k /home/wanghuadong/liangshihao/CoKE-paddle/fb15k
python main.py \
--data_root /home/liangshihao/data/fb15k \
--use_cuda True \
--vocab_size 16396 \
--max_seq_len 3 \
--task_name triple \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--bmtrain False \
--model_name coke \
--checkpoint_num 50 \
--batch_size 2048 \
--learning_rate 1e-3 \
--gpu_ids 1 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain True \
--save_path ./checkpoints/test/ \
--epoch 200

# path data
python main.py \
--data_root /home/wanghuadong/liangshihao/CoKE-paddle/pathqueryFB \
--vocab_size 75169 \
--max_seq_len 7 \
--task_name path \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--model_name coke \
--checkpoint_num 50 \
--batch_size 2048 \
--learning_rate 1e-3 \
--gpu_ids 1 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain True \
--save_path ./checkpoints/test/ \
--epoch 200 \
--bmtrain False
```

```shell
# For BMTrain
MASTER_ADDR='10.31.118.166'
MASTER_PORT='34724'
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
torchrun --standalone --nnodes=1 --nproc_per_node=2 ./main.py \
--data_root /home/liangshihao/data/fb15k \
--use_cuda True \
--vocab_size 16396 \
--max_seq_len 3 \
--task_name triple \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--bmtrain False \
--model_name coke \
--checkpoint_num 50 \
--batch_size 2048 \
--learning_rate 1e-4 \
--gpu_ids 0 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain True \
--save_path ./checkpoints/test/ \
--epoch 200 \
--bmtrain True
```

#### to evaluate

```shell
# triple data
# /home/liangshihao/data/fb15k /home/wanghuadong/liangshihao/CoKE-paddle/fb15k
python evaluation.py \
--task_name triple \
--data_root /home/liangshihao/data/fb15k \
--vocab_size 16396 \
--max_seq_len 3 \
--use_cuda True \
--gpu_ids 0 \
--batch_size 2048 \
--test_file test.coke.txt \
--checkpoint ./checkpoints/coke_lr0.0005_bs2048_step944.pt \
--save_path ./checkpoints/ \
--model_name coke \
--use_ema False

# BMTrain
torchrun --standalone --nnodes=1 --nproc_per_node=1 ./evaluation.py \
--task_name triple \
--data_root /home/liangshihao/data/fb15k \
--vocab_size 16396 \
--max_seq_len 3 \
--use_cuda True \
--gpu_ids 0 \
--batch_size 2048 \
--test_file test.coke.txt \
--checkpoint ./checkpoints/coke_lr0.0005_bs2048_step944.pt \
--save_path ./checkpoints/ \
--model_name coke \
--use_ema False \
--bmtrain True
```

#### todo
roberta loss收敛至5.6左右不再降，valid acc：10%
transformer loss收敛至5.9左右不再降，valid acc：9%
复现论文评估指标, 与bmtrain比较

## For KEPLER

roberta-base文件夹：/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base/

#### requirements

- transformers==4.18.0
- torch==1.11.0

#### forward test

```shell
BMKG/bmkg/models/bigmodel/$
python kepler.py \
--num_classes 2 \
--base_model roberta \
--pooler_dropout 0.2 \
--gamma 1 \
--nrelation 50265 \
--ke_model TransE \
--padding_idx 100
```

