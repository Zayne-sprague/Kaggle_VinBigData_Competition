from torch import optim
from torch.nn import Module
from torch.utils.data import random_split, DataLoader

from typing import List, Optional
import logging
from tqdm import tqdm

from src import training_log, config
from src.models.model import BaseModel
from src.data_loaders.data_loader import TrainingDataLoader
from src.visualizations.visualization import Visualization
from src.utils.hooks import HookBase
from src.utils.events import EventStorage

class TrainingTask:

    def __init__(self):

        self.log: logging.Logger = training_log

        self.start_iter: int = 0
        self.iter: int = 0
        self.max_iter: int = 0

        self.hooks: List[HookBase] = []
        self.eventStorage: Optional[EventStorage] = None

    # def register_training_data(self, data: TrainingDataLoader, train_to_val_split: float = 0.0) -> None:
    #     assert 0.0 <= train_to_val_split <= 1.0, "Please pass in a correct value for train_to_val_split 0 <= x <= 1.0"
    #
    #     self.training_data_loader = data
    #
    #     if train_to_val_split > 0.0:
    #         train_size = int(len(self.training_data_loader) * train_to_val_split)
    #         val_size = int(len(self.training_data_loader) - train_size)
    #         self.train_data, self.val_data = random_split(self.training_data_loader, [train_size, val_size])
    #     else:
    #         self.train_data = self.training_data_loader
    #         self.val_data = None

    def begin_or_resume(self, resume=True):
        if resume:
            self.__resume__()
        self.train()

    def train(self) -> None:
        self.before_training()

        for i in range(self.iter, self.max_iter):
            self.before_iteration()
            self.step()
            self.after_iteration()

        self.after_training()

    def step(self):
        pass

    def __resume__(self) -> None:
        pass

    def before_training(self):
        [x.before_training() for x in self.hooks]

    def after_training(self):
        [x.before_training() for x in self.hooks]

    def before_iteration(self):
        [x.before_iteration() for x in self.hooks]

    def after_iteration(self):
        [x.after_iteration() for x in self.hooks]


# Inspired by detectron2s structure.
class SimpleTrainer(TrainingTask):
    def __init__(self, model: BaseModel, data: TrainingDataLoader, optimizer: optim.Optimizer):
        super().__init__()

        batch_size = config.batch_size

        self.model: BaseModel = model

        self.data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

        self.optimizer: optim.Optimizer = optimizer

