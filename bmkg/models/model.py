import abc
import argparse
import logging
from abc import abstractmethod
from typing import Union, Type, Iterable

import torch
import tqdm
import wandb

from ..data import DataModule


class BMKGModel(abc.ABC, torch.nn.Module):
    step = 0
    epoch = 0
    data_loader: DataModule
    train_data: Iterable
    valid_data: Iterable
    test_data: Iterable
    valid_pbar: tqdm.tqdm
    train_pbar: tqdm.tqdm
    test_pbar: tqdm.tqdm

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.lr = self.config.lr
        self.max_epoch = config.max_epoch
        self.logger = config.logger
        if self.logger == 'wandb':
            wandb.init(
                project="BMKG",
                tags=[config.model],
                entity="leo_test_team",
                config=config
            )
        # TODO: INITIALIZE LOGGER

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    def valid_step(self, *args, **kwargs):
        self.train_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        self.train_step(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def load_data() -> Type[DataModule]:
        pass

    def on_train_start(self) -> None:
        """
        on_train_start hook will be called before train starts.

        by default, we do nothing.
        :return:
        """
        pass

    def on_valid_start(self) -> None:
        """
        on_valid_start hook will be called before validation starts.

        by default, we do nothing.
        :return:
        """
        pass

    def on_valid_end(self) -> None:
        """
        on_valid_end hook will be called after validation ends.

        by default, we do nothing.
        :return:
        """
        pass


    def on_test_start(self) -> None:
        """
        on_test_start hook will be called before test starts.

        by default, we do nothing.
        :return:
        """
        pass

    def on_test_end(self) -> None:
        """
        on_test_end hook will be called after test ends.

        by default, we do nothing.
        :return:
        """
        pass

    def on_epoch_end(self) -> None:
        """
        on_epoch_end hook will be called after each epoch.

        by default, we do nothing.
        :return:
        """
        pass


    def do_train(self, data_loader: DataModule):
        self.train_data = data_loader.train
        self.on_train_start()
        self.train()
        torch.set_grad_enabled(True)
        optim = self.configure_optimizers()
        self.train_pbar = tqdm.tqdm(total=self.max_epoch * len(self.train_data))
        for _ in range(self.max_epoch):
            self.on_epoch_start()
            for data in self.train_data:
                self.step += 1
                loss = self.train_step(data)
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.train_pbar.update(1)
            # TODO: SAVE MODEL
            self.on_epoch_end()
            self.step = 0
            self.epoch += 1
            if self.epoch % self.config.valid_interval == 0:
                self.train_pbar.write("Validating")
                self.do_valid(data_loader)

    @torch.no_grad()
    def do_valid(self, data_loader: DataModule):
        self.valid_data = data_loader.valid
        self.on_valid_start()
        self.eval()
        self.valid_pbar = tqdm.tqdm(total=len(self.valid_data))
        for data in self.valid_data:
            self.step += 1
            self.valid_step(data)
            self.valid_pbar.update(1)
        self.on_valid_end()
        self.train()

    def do_test(self, data_loader: DataModule):
        self.test_data = data_loader.test
        self.on_test_start()
        self.eval()
        torch.set_grad_enabled(False)
        self.test_pbar = tqdm.tqdm(total=len(self.test_data))
        for data in self.test_data:
            self.step += 1
            self.test_step(data)
            self.test_pbar.update(1)
        self.on_test_end()
        self.train()
        torch.set_grad_enabled(True)

    def log(self, key: str, value: Union[int, float, torch.TensorType]):
        if self.logger == 'wandb':
            wandb.log({
                key: value
            })
        # raise NotImplementedError

    def log_hyperparameters(self):
        raise NotImplementedError

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.5, help="Learning rate")
        parser.add_argument('--max_epoch', type=int, default=1, help="How many epochs to run")
        parser.add_argument('--logger', choices=['wandb', 'none'], default='wandb', help="Which logger to use")
        parser.add_argument('--valid_interval', default=1, type=int, help="How many epochs to run before validating")
        return parser
