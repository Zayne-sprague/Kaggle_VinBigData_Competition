import torch
from torch import optim
from torch.utils.data import DataLoader

from typing import List, Optional
import logging
import time

from src import training_log, config
from src.models.model import BaseModel
from src.data.data_set import TrainingDataSet

from src.utils.hooks import HookBase
import weakref

from src.utils.events import EventStorage
from src.training_tasks import BackpropAggregators
from src.data_augs.mix_up import MixUpImage, MixUpImageWithAnnotations

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
    def __init__(self, model: BaseModel, data: TrainingDataSet, optimizer: optim.Optimizer, backward_agg: BackpropAggregators = BackpropAggregators.IndividualBackprops):
        super().__init__()

        batch_size = config.batch_size

        try:
            if config.use_gpu and config.gpu_count > 1:
                self.model = DistributedModel(model, device_ids=config.devices[:-1] if data.__annotated__ else config.devices)
                self.log.info("Distributing model across GPUs")

                self.log.info(f"Using {config.gpu_count} gpus will split the set batch size ({batch_size}) to a batch size of {int(batch_size / config.gpu_count)} per GPU.")
            else:
                self.model: BaseModel = model

            self.model.to(config.devices[0])

        except Exception as e:
            self.log.critical("Could not export the model to the device")
            raise e

        data.display_metrics(data.get_metrics())
        self.data = iter(DataLoader(data, batch_size=batch_size, num_workers=4, collate_fn=collate_fn))

        self.optimizer: optim.Optimizer = optimizer

        self.backward_agg: BackpropAggregators = backward_agg

    def write_iteration_metrics(self, metrics: dict, data_delta: float, inference_delta: float, back_prop_delta:float, step_delta: float):

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
        self.storage.put_item("step_delta", step_delta)

    def step(self):
        assert self.model.training, "Model is in evaluation mode instead of training!"

        data_start = time.perf_counter()
        data: dict = next(self.data)
        for ky, val in data.items():
            # If we can, try to load up the batched data into the device (try to only send what is needed)
            if isinstance(data[ky], torch.Tensor):
                data[ky] = data[ky].to(config.devices[0])

        # TODO - make this an augmenter class, where you register augmenters.
        # Sucks that pytorch doesn't have something similar to this out of the box (augs by batch rather than by ind. samples)
        if 'annotations' in data:
            data = MixUpImageWithAnnotations()(data)
        else:
            data = MixUpImage()(data)

        data_delta = time.perf_counter() - data_start

        inf_start = time.perf_counter()
        loss_dict = self.model(data)['losses']
        inf_delta = time.perf_counter() - inf_start

        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()

        back_prop_start = time.perf_counter()

        if self.backward_agg == BackpropAggregators.IndividualBackprops:
            losses.backward(torch.ones_like(losses))
        elif self.backward_agg == BackpropAggregators.MeanLosses:
            losses.mean().backward()
        else:
            losses.backward()

        self.optimizer.step()
        back_prop_delta = time.perf_counter() - back_prop_start

        self.write_iteration_metrics(loss_dict, data_delta=data_delta, inference_delta=inf_delta, back_prop_delta=back_prop_delta, step_delta=time.perf_counter() - data_start)


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


def collate_fn(batch):
    # TODO - didn't know this was a thing! This is where the data augs per batch should go. Also expand and refactor
    #  this to be a bit better/faster

    image = torch.tensor([item['image'] for item in batch])

    if 'label' in batch[0]:
        label = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        return {'image': image, 'label': label}

    if 'annotations' in batch[0]:
        annotations = [{'boxes': torch.tensor(x['annotations']['boxes']), 'labels': torch.tensor(x['annotations']['labels'], dtype=torch.float)} for x in batch]

        return {'image': image, 'annotations': annotations}