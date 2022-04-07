import argparse
import logging
from typing import Type

from .models import BMKGModel
from .models.transx.transe import TransE
from .models.transx.transr import TransR
from .models.transx.transh import TransH
from .models.transx.transd import TransD

models: dict[str, : Type[BMKGModel]] = {
    'TransE': TransE,'TransR': TransR,'TransH': TransH,'TransD': TransD
}


def main():
    FORMAT = '%(levelname)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    # we first parse the model argument to determine which model to use
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=models.keys(), help="which model to use.")
    conf, model_args = parser.parse_known_args()

    # then we gather arguments for model and dataloader
    model_type = models[conf.model]
    loader_type= model_type.load_data()
    parser = model_type.add_args(parser)
    parser = loader_type.add_args(parser)

    # and finally we parse rest arguments and construct model and data_loader
    config = parser.parse_args()
    data_loader = loader_type(config)
    model: BMKGModel = model_type(config)
    model = model.cuda()
    model.do_train(data_loader)
    if config.eval:
        model.do_valid(data_loader)
    print(config)


if __name__ == '__main__':
    main()
