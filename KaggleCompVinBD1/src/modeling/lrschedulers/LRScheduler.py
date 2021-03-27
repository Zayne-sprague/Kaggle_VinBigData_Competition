import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_warmup as warmup

from typing import Optional

from src.utils.hooks import HookBase

class LRScheduler(HookBase):

    def __init__(self, start_iteration: int = 0, end_iteration: Optional[int] = None):
        super().__init__()
        self.scheduler = None

        self.start_iter = start_iteration
        self.end_iter = end_iteration

    def __register__(self, optimizer: optim.Optimizer):
        self.trainer.log.info("Registering Learning Rate Scheduler")
        self.register(optimizer)

    def register(self, optimizer: optim.Optimizer):
        raise NotImplementedError

    def step(self):
        if not self.scheduler:
            return
        self.scheduler.step()

    def before_iteration(self):
        if self.trainer.iter >= self.start_iter and (not self.end_iter or self.trainer.iter <= self.end_iter) and not self.scheduler:
            self.__register__(self.trainer.optimizer)

    def after_iteration(self):
        if self.trainer.iter == self.end_iter:
            self.scheduler = None


class LambdaLR(LRScheduler):
    def __init__(
            self,
            start_iteration: int = 0,
            end_iteration: Optional[int] = None,
            lambdas=None,
    ):
        super().__init__(start_iteration, end_iteration)
        self.lambdas = lambdas

    def register(self, optimizer: optim.Optimizer):
        self.scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lambdas)


class LinearWarmup(LRScheduler):

    def step(self):
        if not self.scheduler:
            return
        self.scheduler.dampen()

    def register(self, optimizer: optim.Optimizer):
        self.scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.end_iter - self.start_iter)