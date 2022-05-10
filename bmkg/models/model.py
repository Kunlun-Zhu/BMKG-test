import abc
import argparse
import logging
import pathlib
from abc import abstractmethod
from typing import Union, Type, Iterable
from datetime import datetime
import torch
import tqdm
import wandb
import bmtrain as bmt
from ..data import DataModule


class BMKGModel(abc.ABC, bmt.DistributedModule):
    step = 0
    epoch = 0
    data_module: DataModule
    train_data: Iterable
    valid_data: Iterable
    test_data: Iterable
    valid_pbar: tqdm.tqdm
    train_pbar: tqdm.tqdm
    test_pbar: tqdm.tqdm
    scores: list[float]
    score_name: str = "unknown"
    # [ (score, save_path) ]
    best_models: list[(float, pathlib.Path)]

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.config = config
        self.lr = self.config.lr
        self.max_epoch = config.max_epoch
        self.logger = config.logger
        if self.logger == 'wandb':
            if bmt.rank() == 0:
                wandb.init(
                    project="BMKG",
                    tags=[config.model],
                    entity="kunlunz",
                    config=config
                )
        now = datetime.now()
        self.ckpt_path = pathlib.Path(config.ckpt_path) / type(self).__name__ / now.strftime("%Y-%m-%d-%H-%M-%S")
        self.scores = []
        self.fused = config.fused
        self.best_models = []
            
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

    def _save_model(self):
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        save_path = self.ckpt_path / f"epoch-{self.epoch}-{self.score_name}-{self.scores[-1]:.4f}.ckpt"
        # torch.save(self.state_dict(), save_path)
        bmt.save(self, save_path)
        self.best_models.append((self.scores[-1], save_path))

    def save_model(self) -> bool:
        if len(self.scores) == 0:
            return False
        if not self.config.ckpt:
            return False
        if self.config.ckpt_mode == 'all':
            logging.info("Saving model")
            self._save_model()
        elif self.config.ckpt_mode == 'best':
            if len(self.best_models) < self.config.ckpt_best_n:
                logging.info("Saving best models")
                # just save
                self._save_model()
            else:
                smallest_score: tuple[float, pathlib.Path] = min(self.best_models, key=lambda x: x[0])
                if self.scores[-1] > smallest_score[0]:
                    logging.info("Replacing best models")
                    smallest_score[1].unlink()
                    self.best_models.remove(smallest_score)
                    self._save_model()
        if self.config.early_stopping and len(self.scores) >= self.config.early_stopping_tolerance:
            stop = True
            for i in range(-self.config.early_stopping_tolerance, 0):
                if self.scores[i] <= self.scores[i + 1]:
                    stop = False
                    break
            if stop:
                return True
        return False

    def do_train(self, data_module: DataModule):
        try:
            self.train_data = data_module.train
            self.data_module = data_module
            self.on_train_start()
            self.train()
            torch.set_grad_enabled(True)
            optim = self.configure_optimizers()
            #lr_scheduler = bmt.lr_scheduler.Noam(optim, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)
            bmt.synchronize()
            self.train_pbar = tqdm.tqdm(total=self.max_epoch * len(self.train_data), desc="train")
            for _ in range(self.max_epoch):
                self.on_epoch_start()
                for data in self.train_data:
                    self.step += 1
                    optim.zero_grad()
                    with bmt.inspect.inspect_tensor() as inspector:
                        loss = self.train_step(data)
                        if self.fused:
                        #only when using fused optimizer in bmtrain
                            loss = optim.loss_scale(loss)
                        loss.backward()
                        summary = inspector.get_summary()
                        text_summary = bmt.inspect.format_summary(summary)
                        bmt.print_rank(text_summary)
                    bmt.optim_step(optim)
                    #bmt.optim_step(optim, lr_scheduler)
                    self.train_pbar.update(1)
                self.on_epoch_end()
                self.step = 0
                self.epoch += 1
                if self.epoch % self.config.valid_interval == 0:
                    self.train_pbar.write("Validating")
                    self.do_valid(data_module)
                    '''
                    if self.save_model():
                        logging.info(f"Early stopping on {self.epoch=}")
                        return
                    '''
        except KeyboardInterrupt:
            logging.warning("Stopping test!")
            logging.warning("Saving model, just in case")
            logging.warning("Press Ctrl-C Again to force quit")
            self.save_model()

    @torch.no_grad()
    def do_valid(self, data_module: DataModule):
        self.valid_data = data_module.valid
        self.data_module = data_module
        self.on_valid_start()
        self.eval()
        self.valid_pbar = tqdm.tqdm(total=len(self.valid_data), desc="valid")
        for data in self.valid_data:
            self.step += 1
            self.valid_step(data)
            self.valid_pbar.update(1)
        self.on_valid_end()
        self.train()

    def do_test(self, data_module: DataModule):
        if len(self.best_models) != 0:
            path = max(self.best_models, key=lambda x: x[0])[1]
            logging.info(f"Loading best model {path=} to test")
            # self.load_state_dict(torch.load(path))
            bmt.load(self, path)
        self.test_data = data_module.test
        self.data_module = data_module
        self.on_test_start()
        self.eval()
        torch.set_grad_enabled(False)
        self.test_pbar = tqdm.tqdm(total=len(self.test_data), desc="test")
        for data in self.test_data:
            self.step += 1
            self.test_step(data)
            self.test_pbar.update(1)
        self.on_test_end()
        self.train()
        torch.set_grad_enabled(True)

    def log(self, key: str, value: Union[int, float, torch.TensorType]):
        if bmt.rank() == 0:
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
        parser.add_argument('--valid_interval', default=1, type=int,
                            help="How many epochs to run before validating"
                                 "Note that this parameter will effect early stopping")
        parser.add_argument('--ckpt_path', default="./data/ckpt", help="Checkpoint path")
        parser.add_argument('--early_stopping', default=True,
                            help="Whether to stop training if the performance isn't improving")
        parser.add_argument('--early_stopping_tolerance', default=3, type=int,
                            help="Number of epochs to wait before early stopping")
        parser.add_argument('--not_ckpt', dest="ckpt", default=True, action="store_false",
                            help="Don't save check points")
        parser.add_argument('--ckpt_mode', choices=['all', 'best'], default='best',
                            help="Whether to save all models or only best n models.")
        parser.add_argument('--ckpt_best_n', default=3, type=int,
                            help="How many best models to save when `ckpt_model` is `best`.")
        return parser
