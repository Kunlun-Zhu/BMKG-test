import argparse
import logging
import torch
from typing import Type

from .models import BMKGModel
from .models.transx.transe import TransE
import bmtrain as bmt

models: dict[str, : Type[BMKGModel]] = {
    'TransE': TransE
}


def main():
    bmt.init_distributed(seed=0)
    print(torch.cuda.is_available())
    FORMAT = '%(levelname)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    # we first parse the model argument to determine which model to use
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=models.keys(), help="which model to use.")
    parser.add_argument('--not_train', dest="train", default=True, action="store_false", help="Not train")
    parser.add_argument('--test', dest="test", default=False, action="store_true", help="Test the model")
    conf, _ = parser.parse_known_args()

    # then we gather arguments for model and dataloader
    model_type = models[conf.model]
    loader_type= model_type.load_data()
    parser = model_type.add_args(parser)
    parser = loader_type.add_args(parser)

    # and finally we parse rest arguments and construct model and data_module
    config = parser.parse_args()
    data_module = loader_type(config)
    model: BMKGModel = model_type(config)
    model = model.cuda()
    #init model parameters using bmt.init_parameters
    bmt.init_parameters(model)
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()
    model.do_train(data_module)
    if conf.test:
        model.do_test(data_module)
    #bmt save models       
    bmt.save(model, "checkpoint_" + conf.model  + ".pt")


if __name__ == '__main__':
    main()
