from torch import optim
from torch.nn import Module
from torch.utils.data import random_split, DataLoader

from typing import List, Optional
import logging
from tqdm import tqdm

from src import training_log
from src.models.model import BaseModel
from src.data_loaders.data_loader import TrainingDataLoader
from src.visualizations.visualization import Visualization


class TrainingTask:

    def __init__(
            self,
            task_name: str,

            learning_rate: float = 0.0001,

            epochs: int = 10,
            batch_size: int = 16,

            checkpoint_frequency: int = 0,
            validation_frequency: int = 0,
             ):

        self.log: logging.Logger = training_log

        self.task_name: str = task_name

        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[Module] = None

        self.training_data_loader: Optional[TrainingDataLoader] = None
        self.train_data: Optional[TrainingDataLoader] = None
        self.val_data: Optional[TrainingDataLoader] = None

        self.checkpoint_frequency: int = checkpoint_frequency
        self.validation_frequency: int = validation_frequency

        self.__training_losses__: List[float] = []
        self.__validation_losses__: List[float] = []

        self.__state__: dict = {}

        self.__visualization__: Visualization = Visualization()

    def register_training_data(self, data: TrainingDataLoader, train_to_val_split: float = 0.0) -> None:
        assert 0.0 <= train_to_val_split <= 1.0, "Please pass in a correct value for train_to_val_split 0 <= x <= 1.0"

        self.training_data_loader = data

        if train_to_val_split > 0.0:
            train_size = int(len(self.training_data_loader) * train_to_val_split)
            val_size = int(len(self.training_data_loader) - train_size)
            self.train_data, self.val_data = random_split(self.training_data_loader, [train_size, val_size])
        else:
            self.train_data = self.training_data_loader
            self.val_data = None

    def validate(self, model: BaseModel) -> Optional[float]:
        assert self.val_data and len(self.val_data) > 0, "Please set the validation dataset of the model before validating"

        self.log.info(f'Beginning Evaluation of model')

        loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

        p_bar = tqdm(loader, desc=f'Evaluating: 0.00, Number Correct: 0', total=len(loader))

        total_loss = 0

        for batch in p_bar:
            loss = self.__inner_validation_loop__(model, batch, self.__optimizer__(model), self.__criterion__(model))
            total_loss += loss

            p_bar.set_description(f'Evaluating: {loss:.2f}')

        self.log.info(f'Ending evaluation of the model. Total loss = {total_loss / len(self.val_data):.2f}')

        return total_loss / len(self.val_data)

    def begin_or_resume(self, model: BaseModel, resume=True):
        if resume:
            self.__resume__(model)
        self.__training_job__(model)

    def __training_job__(self, model) -> None:

        start = self.__state__.get("epoch", 0)
        end = self.epochs

        for i in range(start, end):
            epoch_loss: float = self.__epoch__(model, epoch=i)
            val_loss: Optional[float] = None

            if self.checkpoint_frequency > 0 and 0 == (i+1) % self.checkpoint_frequency:
                self.__checkpoint__(model=model)
            if self.validation_frequency > 0 and 0 == (i+1) % self.validation_frequency:
                val_loss = self.validate(model)

                if len(self.__validation_losses__) - len(self.__training_losses__) > 0:
                    self.__validation_losses__ += [val_loss] * (len(self.__validation_losses__) - len(self.__training_losses__))

            self.__training_losses__.append(epoch_loss)

            if val_loss:
                self.__validation_losses__.append(val_loss)

        # Reset
        self.optimizer = None
        self.criterion = None

    def __epoch__(self, model: BaseModel, epoch: int, *args, **kwargs) -> float:
        self.log.info(f'Beginning epoch {epoch + 1} for training')

        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

        p_bar = tqdm(loader, desc=f'Training Epoch {epoch + 1} : Loss 0', total=len(loader))

        total_epoch_loss: float = 0.0
        for batch in p_bar:
            loss = self.__inner_training_loop__(model, batch, self.__optimizer__(model), self.__criterion__(model))
            self.__visualization__.add_data([loss])
            total_epoch_loss += loss

            p_bar.set_description(f'Training Epoch {epoch + 1} : Loss {loss:.2f}')

        self.log.info(f'Epoch {epoch + 1} finished with a loss of {total_epoch_loss / len(self.train_data):.2f}')

        return total_epoch_loss / len(self.train_data)

    def __checkpoint__(self, model: BaseModel) -> None:
        raise NotImplementedError()

    def __resume__(self, model: BaseModel) -> None:
        raise NotImplementedError()

    def __inner_training_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        raise NotImplementedError()

    def __inner_validation_loop__(self, model: Module, batch: dict, optimizer: optim.Optimizer, criterion: Module, *args, **kwargs) -> float:
        raise NotImplementedError()

    def __optimizer__(self, model: BaseModel) -> optim.Optimizer:
        raise NotImplementedError()

    def __criterion__(self, model: BaseModel) -> Module:
        raise NotImplementedError()