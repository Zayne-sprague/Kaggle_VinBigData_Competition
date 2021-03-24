import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, BufferedShuffleDataset

from ignite.engine import Engine, Events

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as E
from pytorch_pfn_extras.training.extension import Extension, PRIORITY_READER
from pytorch_pfn_extras.training.manager import ExtensionsManager
from pytorch_pfn_extras.training import IgniteExtensionsManager

from sklearn.model_selection import StratifiedKFold
import timm
from typing import Mapping, Any
import albumentations as A
import numpy as np

from src.data.abnormal_dataset import TrainingAbnormalDataSet
from src.data.data_set import TrainingDatasetMixin
from src.utils.paths import MODELS_DIR
from src import config, log

dataset = TrainingAbnormalDataSet(image_transformations=A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=25, max_width=25, p=0.5),
    A.Blur(blur_limit=[3, 7], p=0.5),
    A.Downscale(scale_min=0.25, scale_max=0.9, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.6)
]))

dataset.load_records()

dataset = TrainingDatasetMixin(dataset)


class CNNFixedPredictor(nn.Module):
    def __init__(self, cnn: nn.Module, num_classes: int = 2):
        super(CNNFixedPredictor, self).__init__()
        self.cnn = cnn
        self.lin = nn.Linear(cnn.num_features, num_classes)

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.cnn(x)
        return self.lin(feat)


def build_predictor(model_name: str, model_mode: str = 'normal'):

    if model_mode == 'normal':
        return timm.create_model(model_name, pretrained=True, num_classes=2, in_chans=3)
    elif model_name == 'cnn_fixed':
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=3)
        return CNNFixedPredictor(timm_model, num_classes=2)

def accuracy(y: torch.Tensor, t: torch.Tensor):
    assert y.shape[:-1] == t.shape
    pred_label = torch.max(y.detach(), dim=-1)[1]
    count = t.nelement()
    correct = (pred_label == t).sum().float()
    acc = correct / count
    return acc

def accuracy_with_logits(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    assert y.shape == t.shape
    gt_label = torch.max(t.detach(), dim=-1)[1]
    return accuracy(y, gt_label)

def cross_entropy_with_logits(inp, tar, dim=-1):
    loss = torch.sum(-tar * F.log_softmax(inp, dim), dim)
    return loss.mean()


class Classifier(nn.Module):

    def __init__(self, predictor, lossfun=cross_entropy_with_logits):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix=''

    def forward(self, image, targets):
        outputs = self.predictor(image.float())
        loss = self.lossfun(outputs, targets)
        metrics = {
            f'{self.prefix}loss': loss.item(),
            f'{self.prefix}acc': accuracy_with_logits(outputs, targets).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics

    def predict(self, data_loader):
        pred = self.predict_proba(data_loader)
        label = torch.argmax(pred, dim=1)
        return label

    def predict_proba(self, data_loader):
        device = config.devices[0]

        y_list = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0].to(device)
                else:
                    batch = batch.to(device)
                y = self.predictor(batch)
                y = torch.softmax(y, dim=-1)
                y_list.append(y)
        pred = torch.cat(y_list)
        return pred


class EMA:

    def __init__(
            self,
            model: nn.Module,
            decay: float,
            strict: bool = True,
            use_dynamic_decay: bool = True
    ):
        self.decay = decay
        self.model = model
        self.strict = strict
        self.use_dynamic_decay = use_dynamic_decay
        self.logger = log
        self.n_step = 0

        self.shadow = {}
        self.original = {}

        self._assigned = False

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def step(self):
        self.n_step ++ 1
        if self.use_dynamic_decay:
            _n_step = float(self.n_step)
            decay = min(self.decay, (1.0 + _n_step) / (10.0 + _n_step))
        else:
            decay = self.decay

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    __call__ = step

    def assign(self):
        if self._assigned:
            if self.strict:
                raise ValueError("Error assign is called again before resume")
            else:
                self.logger.warning(
                    "`assign` is called again before `resume`."
                    "shadow parameter is already assigned, skip."
                )
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
        self._assigned = True

    def resume(self):
        """Restore original parameters to a model.

        That is, put back the values that were in each parameter at the last call to `assign`.
        """
        if not self._assigned:
            if self.strict:
                raise ValueError("[ERROR] `resume` is called before `assign`.")
            else:
                self.logger.warning("`resume` is called before `assign`, skip.")
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
        self._assigned = False


class LRScheduler(Extension):

    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    name = None

    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str, scheduler_kwargs: Mapping[str, Any]):
        super().__init__()
        self.scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_kwargs)

    def __call__(self, manager: ExtensionsManager):
        self.scheduler.step()

    def state_dict(self) -> None:
        return self.scheduler.state_dict()

    def load_state_dict(self, to_load) -> None:
        self.scheduler.load_state_dict(to_load)


def create_trainer(model, optimizer) -> Engine:
    device = config.devices[0]
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(batch['image'].to(config.devices[0]), batch['label'].to(config.devices[0]))
        loss.backward()
        optimizer.step()
        return metrics

    trainer = Engine(update_fn)
    return trainer


skf = StratifiedKFold(n_splits=5, shuffle=True)
y = np.array([int(x['label'][0] < x['label'][1]) for x in dataset.records])
split_inds = list(skf.split(dataset, y))
train_inds, valid_inds = split_inds[0]

train_loader = DataLoader(
    [dataset[i] for i in train_inds],
    batch_size=config.batch_size,
    num_workers=0 if config.use_gpu else 0,
    pin_memory=True
)

valid_loader = DataLoader(
    [dataset[i] for i in valid_inds],
    batch_size=config.batch_size,
    num_workers=0 if config.use_gpu else 0,
    pin_memory=True
)

predictor = build_predictor(model_name='resnet18', model_mode='normal')
classifier = Classifier(predictor)
model = classifier

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-3)

trainer = create_trainer(model, optimizer)

ema = EMA(predictor, decay=0.999)

def eval_func(**batch):
    loss, metrics = model(batch['image'].to(config.devices[0]), batch['label'].to(config.devices[0]))

    classifier.prefix ='ema_'
    ema.assign()
    loss, metrics = model(batch['image'].to(config.devices[0]), batch['label'].to(config.devices[0]))
    ema.resume()
    classifier.prefix = ''

valid_evaluator = E.Evaluator(
    valid_loader, model, progress_bar=False, eval_func=eval_func, device=config.devices[0]
)

log_trigger = (1, 'epoch')
log_report = E.LogReport(trigger=log_trigger)
extensions = [
    log_report,
    E.ProgressBar(update_interval=10 if config.DEBUG else 100),
    E.PrintReport(),
    E.FailOnNonNumber(),
]

epoch = 20
models = {"main": model}
optimizers = {"main": optimizer}
manager = IgniteExtensionsManager(
    trainer, models, optimizers, epoch, extensions=extensions, out_dir=str(MODELS_DIR / 'note_book_2_classifier_ex')
)

manager.extend(valid_evaluator)

manager.extend(
    E.snapshot_object(predictor, 'predictor.pt'), trigger=(5, 'epoch')
)

manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

# Exponential moving average
manager.extend(lambda manager: ema(), trigger=(1, "iteration"))

def save_ema_model(manager):
    ema.assign()
    torch.save(predictor.state_dict(), MODELS_DIR / "predictor_ema.pt")
    ema.resume()

manager.extend(save_ema_model, trigger=(5, "epoch"))

_ = trainer.run(train_loader, max_epochs=epoch)

torch.save(predictor.state_dict(), MODELS_DIR / "2ClassClassifier.pt")
