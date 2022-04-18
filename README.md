## Data部分的设计

数据流： 由模型返回DataLoader，DataLoader 根据命令行参数构造DataSet，从DataSet拿到DataBatch后传递给Sampler， 
Sampler将pos和neg的DataBatch传递给模型。

# Guid for training TransX models and semantic models


To get started, python version 3.9, and maturin is required.

In order to run successfully, around 2.5G memory is required in GPU.

After installing this branch by using git clone.



```shell
# You should first Develop the project with the command
maturin develop -r

# And then preprocess the data with the command
python3 -m bmkg.preprocess --data_path ${'DATA_PATH'} --data_files train.txt valid.txt test.txt

# Finally you can run the training process with the command
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --model ${'MODEL_NAME'} --test --fused --optim Bmtrain --data_path ${'DATA_PATH'} --data_files train.npy valid.npy test.npy

```

Current supported models are:
- [x] TransE
- [x] TransR
- [x] TransD
- [x] TransH
- [x] Analogy
- [x] Complex
- [x] Distmult
- [ ] Hole
- [x] Rescal
- [x] Rotate
- [x] Simple
