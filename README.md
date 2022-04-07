## Data部分的设计

数据流： 由模型返回DataLoader，DataLoader 根据命令行参数构造DataSet，从DataSet拿到DataBatch后传递给Sampler， 
Sampler将pos和neg的DataBatch传递给模型。