import torch
from torch import nn
from torchvision.models import resnet50

from src.models.model import BaseModel
from src.utils.hooks import CheckpointHook


class Res50(BaseModel):
    def __init__(self):
        super().__init__(model_name="Res50")

        self.model = nn.Sequential(*(list(resnet50(True).children()))[:-1])

        self.fc = nn.Linear(2048, 2)
        self.lsoft = nn.LogSoftmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data: dict) -> dict:
        x = data['image']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        x = self.model(x)

        x = x.view([batch_size, -1])
        x = self.fc(x)
        predictions = self.lsoft(x)

        out = {'preds': predictions}

        if self.training:
            out['losses'] = self.loss(out, data)

        return out

    def loss(self, predictions: dict, data: dict) -> dict:
        predictions: torch.Tensor = predictions['preds']
        loss = self.criterion(predictions, data['label'])
        return {'loss': loss}


class ResnetCheckpointHook(CheckpointHook):

    def build_state(self) -> dict:
        assert self.trainer.__getattribute__('model') is not None, 'trainer does not have the model to checkpoint'

        model: Res50 = self.trainer.model
        assert isinstance(model, Res50), 'Model found in trainer is NOT the resnet50 model'

        state = {}

        optimizer: optim.Optimizer = self.trainer.optimizer
        state['optim_state'] = optimizer.state_dict()

        state['iteration'] = self.trainer.iter

        return state

    def on_resume(self):
        state = super().on_resume()

        if 'optim_state' in state:
            self.trainer.optimizer.load_state_dict(state['optim_state'])

        if 'iteration' in state:
            self.trainer.start_iter = state['iteration']
            self.trainer.iter = state['iteration']


if __name__ == "__main__":
    from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, TrainingVisualizationHook, \
        LogTrainingLoss
    from torch import optim

    from src.training_tasks import BackpropAggregators

    model = Res50()

    dataloader = TrainingAbnormalDataLoader()
    dataloader.load_records(keep_annotations=False)

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingAbnormalDataLoader)

    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses)
    task.max_iter = 2500

    # val_hook = PeriodicStepFuncHook(250, lambda: task.validation(val_dl, model))
    # checkpoint_hook = ResnetCheckpointHook(250, "resnet50_test2")

    task.register_hook(LogTrainingLoss())
    task.register_hook(StepTimer())
    # task.register_hook(val_hook)
    # task.register_hook(checkpoint_hook)

    task.begin_or_resume()
