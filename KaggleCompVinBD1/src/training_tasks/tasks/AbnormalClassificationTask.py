import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import random_split, Dataset, DataLoader
from torch import optim

from src.models.model import BaseModel
from src.training_tasks.training_task import TrainingTask


class AbnormalClassificationTask(TrainingTask):

    def __init__(
            self,
            task_name: str,

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

    def __checkpoint__(self, model: BaseModel) -> None:
        model.checkpoint(name=self.task_name)

    def __resume__(self, model: BaseModel) -> None:
        state = model.load(name=self.task_name)

        # TODO - implement this further so that the model picks up where it left off.
        # self.__state__['epochs'] = state['epochs']

    def __inner_training_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        x: torch.Tensor = batch['image']
        y: torch.Tensor = batch['label']

        optimizer.zero_grad()
        predictions = model(x)

        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        return loss.item()

    def __inner_validation_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        x: torch.Tensor = batch['image']
        y: torch.Tensor = batch['label']

        predictions = model(x)

        loss = criterion(predictions, y).item()

        return loss

    def __optimizer__(self, model: BaseModel) -> optim.Optimizer:
        if not self.optimizer:
            self.optimizer = optim.Adam(model.parameters(), self.learning_rate)
        return self.optimizer

    def __criterion__(self, model: BaseModel) -> Module:
        if not self.criterion:
            self.criterion = nn.NLLLoss()
        return self.criterion
