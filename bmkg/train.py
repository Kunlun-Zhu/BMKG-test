import argparse
import logging
from typing import Type
import os.path
import torch

from .models import BMKGModel
from .models.transx.transe import TransE
from .models.transx.transr import TransR
from .models.transx.transh import TransH
from .models.transx.transd import TransD
from .models.semantic.analogy import Analogy
from .models.semantic.complex import ComplEx
from .models.semantic.distmult import DistMult
from .models.semantic.hole import HolE
from .models.semantic.rescal import RESCAL
from .models.semantic.rotate import RotatE
from .models.semantic.simple import SimplE

models: dict[str, : Type[BMKGModel]] = {
    'TransE': TransE,
    'TransR': TransR,
    'TransH': TransH,
    'TransD': TransD,
    'Analogy': Analogy,
    'ComplEx': ComplEx,
    'DistMult': DistMult,
    'HolE': HolE,
    'RESCAL': RESCAL,
    'RotatE': RotatE,
    'SimplE': SimplE

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

    path = os.getcwd()
    #if not os.path.exists(path + '\\saved_models'):
    #    print(path + '\\saved_models')
    #    os.makedirs('saved_models')
    #if not os.path.exists(path + '\\saved_models\\begin'):
    #    os.makedirs('saved_models/begin')

    torch.save(model.state_dict, path + '\\saved_models' + '\\begin' + '\\' + conf.model + '.pt')
    model.do_train(data_loader)
    if config.eval:
        model.do_valid(data_loader)
    print(config)
    torch.save(model.state_dict, path + '\\saved_models' + '\\' + conf.model + '.pt')


if __name__ == '__main__':
    main()
