import torch
from torch import optim
from torch.utils.data import DataLoader, BufferedShuffleDataset

from typing import List, Optional
import logging
import time

from src import training_log, config
from src.modeling.models.model import BaseModel
from src.data.data_set import TrainingDataSet

from src.utils.hooks import HookBase
import weakref

from src.utils.events import EventStorage
from src.utils.collater import Collater, SimpleCollater
from src.training_tasks import BackpropAggregators
from src.data_augs.batch_augmenter import BatchAugmenter

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
            self.resume()
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

    def register_hook(self, hook: HookBase):
        hook.trainer = weakref.proxy(self)
        self.hooks.append(hook)

    def resume(self) -> None:
        [x.on_resume() for x in self.hooks]

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
    def __init__(
            self,
            model: BaseModel,
            data: TrainingDataSet,
            optimizer: optim.Optimizer,
            backward_agg: BackpropAggregators = BackpropAggregators.IndividualBackprops,
            batch_augmenter: BatchAugmenter = None,
            collater: Collater = None
    ):

        super().__init__()

        batch_size = config.batch_size

        try:
            if config.use_gpu and config.gpu_count >= 1:
                self.model = DistributedModel(model, device_ids=config.devices)
                self.log.info("Distributing model across GPUs")

                self.log.info(f"Using {config.gpu_count} gpus will split the set batch size ({batch_size}) to a batch size of {int(batch_size / config.gpu_count)} per GPU.")
            else:
                self.model: BaseModel = model

            self.model.to(config.devices[0])

        except Exception as e:
            self.log.critical("Could not export the model to the device")
            raise e

        data.display_metrics(data.get_metrics())

        if collater and batch_augmenter:
            self.log.warning("Both Collater and Batch Augmenter passed into training task-- defaulting to the Collater for batch augmentations (ignoring the batch aug passed in)")
        if not collater:
            collater = SimpleCollater(batch_augmenter=batch_augmenter)

        self.collater: Collater = collater

        self.data = iter(DataLoader(
            BufferedShuffleDataset(data, buffer_size=2500),
            batch_size=batch_size,
            num_workers=4 if config.use_gpu else 0,
            collate_fn=self.collater,
        ))

        self.optimizer: optim.Optimizer = optimizer

        self.backward_agg: BackpropAggregators = backward_agg

    def write_iteration_metrics(self, metrics: dict, data_delta: float, inference_delta: float, back_prop_delta:float, optim_delta:float, step_delta: float, other_metrics: dict):

        for key in metrics:
            dims = metrics[key].shape
            if len(dims) > 0:
                item_list = metrics[key].tolist()
                [self.storage.put_item(key, x) for x in item_list]
            else:
                self.storage.put_item(key, metrics[key].item())


        self.storage.put_item("data_delta", data_delta)
        self.storage.put_item("inference_delta", inference_delta)
        self.storage.put_item("back_prop_delta", back_prop_delta)
        self.storage.put_item("optim_delta", optim_delta)
        self.storage.put_item("step_delta", step_delta)

        for ky in other_metrics:
            self.storage.put_item(ky, other_metrics[ky])

    def step(self):
        assert self.model.training, "Model is in evaluation mode instead of training!"

        step_start = time.perf_counter()

        iters = (config.artificial_batch_size // config.batch_size)

        _losses = []
        data_deltas = []
        inf_deltas = []
        back_prop_deltas = []
        other_metrics = {}

        i = 0
        while i < iters:

            data_start = time.perf_counter()
            data: dict = next(self.data)
            for ky, val in data.items():
                # If we can, try to load up the batched data into the device (try to only send what is needed)
                if isinstance(data[ky], torch.Tensor):
                    data[ky] = data[ky].to(config.devices[0])

            dd = time.perf_counter() - data_start


            inf_start = time.perf_counter()
            loss_dict = self.model(data)

            if 'error' in loss_dict and loss_dict['error']:
                continue

            om = loss_dict.get("other_metrics", {})
            loss_dict = loss_dict['losses']
            inf_deltas.append(time.perf_counter() - inf_start)
            data_deltas.append(dd)

            for ky in om:
                if ky not in other_metrics:
                    other_metrics[ky] = []
                other_metrics[ky].append(om[ky])


            losses = sum(loss_dict.values())
            _losses.append(losses)

            back_prop_start = time.perf_counter()

            if self.backward_agg == BackpropAggregators.IndividualBackprops:
                losses.backward(torch.ones_like(losses))
            elif self.backward_agg == BackpropAggregators.MeanLosses:
                losses.mean().backward()
            else:
                losses.backward()
            back_prop_deltas.append(time.perf_counter() - back_prop_start)

            i += 1

        optim_step_start = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad()
        optim_delta = time.perf_counter() - optim_step_start

        data_delta = sum(data_deltas) / len(data_deltas)
        inf_delta = sum(inf_deltas) / len(inf_deltas)
        back_prop_delta = sum(back_prop_deltas) / len(back_prop_deltas)

        loss_dict = {'loss': torch.tensor(_losses).mean()}

        for ky in other_metrics:
            other_metrics[ky] = sum(other_metrics[ky]) / len(other_metrics[ky])

        self.write_iteration_metrics(loss_dict, data_delta=data_delta, inference_delta=inf_delta, back_prop_delta=back_prop_delta, optim_delta=optim_delta, step_delta=time.perf_counter() - step_start, other_metrics=other_metrics)


class DistributedModel(torch.nn.DataParallel):
    """
    Wrapper for DataParellel so you can reference values from the base model object even though we are distributing it
    """
    def __init__(self, module: torch.nn.Module, device_ids):
        super().__init__(module, device_ids=device_ids)

    def __getattr__(self, name):
        #https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


