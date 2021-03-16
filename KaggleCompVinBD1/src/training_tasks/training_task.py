from torch import optim
from torch.utils.data import DataLoader

from typing import List, Optional
import logging
import time

from src import training_log, config
from src.models.model import BaseModel
from src.data_loaders.data_loader import TrainingDataLoader

from src.utils.hooks import HookBase
import weakref

from src.utils.events import EventStorage


class TrainingTask:

    def __init__(self):

        self.log: logging.Logger = training_log

        self.start_iter: int = 0
        self.iter: int = 0
        self.max_iter: int = 1000

        self.hooks: List[HookBase] = []
        self.storage: Optional[EventStorage] = None

    def begin_or_resume(self, resume=True):
        if resume:
            self.__resume__()
        self.train()

    def train(self) -> None:
        self.before_training()

        with EventStorage() as self.storage:
            for i in range(self.start_iter, self.max_iter):
                self.before_iteration()
                self.step()
                self.after_iteration()

                self.iter = i+1


            self.after_training()

    def step(self):
        pass

    def __resume__(self) -> None:
        pass

    def register_hook(self, hook: HookBase):
        hook.trainer = weakref.proxy(self)
        self.hooks.append(hook)

    def before_training(self):
        [x.before_training() for x in self.hooks]

    def after_training(self):
        [x.after_training() for x in self.hooks]

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

        data.display_metrics(data.get_metrics())
        self.data = iter(DataLoader(data, batch_size=batch_size, num_workers=4))

        self.optimizer: optim.Optimizer = optimizer

    def write_iteration_metrics(self, metrics: dict, data_delta: float, inference_delta: float, back_prop_delta:float, step_delta: float):

        for key in metrics:
            self.storage.put_item(key, metrics[key].item())

        self.storage.put_item("data_delta", data_delta)
        self.storage.put_item("inference_delta", inference_delta)
        self.storage.put_item("back_prop_delta", back_prop_delta)
        self.storage.put_item("step_delta", step_delta)


    def step(self):
        assert self.model.training, "Model is in evaluation mode instead of training!"

        data_start = time.perf_counter()
        data = next(self.data)
        data_delta = time.perf_counter() - data_start

        inf_start = time.perf_counter()
        loss_dict = self.model(data)['losses']
        inf_delta = time.perf_counter() - inf_start

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()

        back_prop_start = time.perf_counter()
        losses.backward()
        self.optimizer.step()
        back_prop_delta = time.perf_counter() - back_prop_start

        self.write_iteration_metrics(loss_dict, data_delta=data_delta, inference_delta=inf_delta, back_prop_delta=back_prop_delta, step_delta=time.perf_counter() - data_start)

