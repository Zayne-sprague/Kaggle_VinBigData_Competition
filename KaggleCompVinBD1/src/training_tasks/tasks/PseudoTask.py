import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import random_split, Dataset, DataLoader
from torch import optim
import time
import random

from src.models.model import BaseModel
from src.training_tasks.training_task import TrainingTask


class PseduoTask(TrainingTask):

    def __init__(
            self,
            task_name: str,

            pseudo_inference_time: float = 0.01,
            pseudo_training_time: float = 0.02,

            learning_rate: float = 0.0001,

            epochs: int = 10,
            batch_size: int = 16,

            checkpoint_frequency: int = 0,
            validation_frequency: int = 0,
    ):
        super().__init__(
            task_name=task_name,

            learning_rate=learning_rate,

            epochs=epochs,
            batch_size=batch_size,

            checkpoint_frequency=checkpoint_frequency,
            validation_frequency=validation_frequency
        )

        self.pseudo_inference_time: float = pseudo_inference_time
        self.pseudo_training_time: float = pseudo_training_time

    def __checkpoint__(self, model: BaseModel) -> None:
        self.log.info("Saving...")

    def __resume__(self, model: BaseModel) -> None:
        self.log.info("Loading...")

    def __inner_training_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        time.sleep(self.pseudo_training_time)
        return random.random()

    def __inner_validation_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        time.sleep(self.pseudo_inference_time)
        return random.random()

    def __optimizer__(self, model: BaseModel) -> optim.Optimizer:
        if not self.optimizer:
            self.optimizer = optim.Adam(model.parameters(), self.learning_rate)
        return self.optimizer

    def __criterion__(self, model: BaseModel) -> Module:
        if not self.criterion:
            self.criterion = nn.NLLLoss()
        return self.criterion
