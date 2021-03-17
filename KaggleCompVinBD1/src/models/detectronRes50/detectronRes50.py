import torch
from torch import nn
from torchvision.models import resnet50

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model

from src.models.model import BaseModel
from src.losses.NLLLossOHE import NLLLossOHE
from src.utils.hooks import CheckpointHook
from src.utils.paths import DATA


class DetectronRes50(BaseModel):
    def __init__(self):
        super().__init__(model_name="Res50")

        trained_weights = str(DATA / 'vbd_r50fpn3x_512px/model_final.pth')

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = trained_weights

        det_model = build_model(cfg)

        self.model = det_model.backbone

        self.f_p2 = nn.Linear(1048576, 1000)
        self.f_p3 = nn.Linear(262144, 1000)
        self.f_p4 = nn.Linear(65536, 1000)
        self.f_p5 = nn.Linear(16384, 1000)
        self.f_p6 = nn.Linear(4096, 1000)

        self.fc = nn.Linear(5000, 2)
        self.lsoft = nn.LogSoftmax(dim=-1)

        self.criterion = NLLLossOHE()

    def forward(self, data: dict) -> dict:
        x = data['image']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        x = self.model(x)

        # TODO - this is probably not correct
        p2 = self.f_p2(x['p2'].view([batch_size, -1]))
        p3 = self.f_p3(x['p3'].view([batch_size, -1]))
        p4 = self.f_p4(x['p4'].view([batch_size, -1]))
        p5 = self.f_p5(x['p5'].view([batch_size, -1]))
        p6 = self.f_p6(x['p6'].view([batch_size, -1]))

        p = torch.stack([p2, p3, p4, p5, p6], dim=2).view([batch_size, -1])
        # TODO - revise the p2-6 mutations for transfer learning (pretty sure each should be making their own pred

        x = self.fc(p)

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
    from src.data.abnormal_dataset import TrainingAbnormalDataSet
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, TrainingVisualizationHook, \
        LogTrainingLoss
    from torch import optim

    from src.training_tasks import BackpropAggregators

    model = DetectronRes50()

    dataloader = TrainingAbnormalDataSet()
    dataloader.load_records(keep_annotations=False)

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingAbnormalDataSet)

    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses)
    task.max_iter = 25000

    val_hook = PeriodicStepFuncHook(5000, lambda: task.validation(val_dl, model))
    checkpoint_hook = ResnetCheckpointHook(1000, "detectron_resnet50_test3")

    task.register_hook(LogTrainingLoss())
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(checkpoint_hook)

    task.begin_or_resume()
