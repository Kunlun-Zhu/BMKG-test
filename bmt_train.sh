#! /bin/bash

export LOCAL_RANK=0
MASTER_ADDR='103.242.175.227'
MASTER_PORT='22'
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ./main.py \
--data_root /home/wanghuadong/liangshihao/CoKE-paddle/fb15k \
--epoch 400 \
--use_cuda True \
--batch_size 512 \
--learning_rate 5e-4 \
--vocab_size 16396 \
--max_seq_len 3 \
--task_name triple \
--checkpoint_num 50 \
--save_path ./checkpoints/ema/ \
--model_name coke \
--gpu_ids 7 \
--soft_label False \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--bmtrain True"
echo ${CMD}
