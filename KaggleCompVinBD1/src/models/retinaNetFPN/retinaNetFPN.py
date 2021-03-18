import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops import sigmoid_focal_loss

from src.models.model import BaseModel
from src.models.res50.res50 import Res50
from src.losses.NLLLossOHE import NLLLossOHE
from src.utils.hooks import CheckpointHook
from src.utils.paths import MODELS_DIR


class RetinaNetFPN(BaseModel):
    def __init__(self):
        super().__init__(model_name="RetinaNetFPN")

        self.m = retinanet_resnet50_fpn(False, num_classes=2)
        # a = Res50()
        # a.load_state_dict(torch.load(f'{MODELS_DIR}/resnet50_test2.pth')['model_state_dict'])


        # self.m.backbone.body.layer1 = self.a.layer1
        # self.m.backbone.body.layer2 = self.a.layer2
        # self.m.backbone.body.layer3 = self.a.layer3
        # self.m.backbone.body.layer4 = self.a.layer4
        # self.m = nn.Sequential(*(list(self.m.children()))[:])

        # Duck patch! worst code I've ever written-- I am so sorry...
        # TODO - find a better way to do this...
        self.m.head.classification_head.compute_loss = lambda *args, **kwargs: class_retina_head_loss(self.m.head.classification_head, *args, **kwargs)

    def forward(self, data: dict) -> dict:
        x = data['image']
        annotations = data['annotations']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        # TODO - we need to remove this to allow for mixup
        # for idx, annotation in enumerate(annotations):
        #     annotations[idx]['labels'] = torch.argmax(annotation['labels'], 1)
        x = self.m(x, annotations)

        if self.training:
            loss = (x['classification'] + x['bbox_regression']).mean()
            out = {'losses': {'loss': loss}}

            return out
        else:
            return x

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

def class_retina_head_loss(self, targets, head_outputs, matched_idxs):
    def _sum(x):
        res = x[0]
        for i in x[1:]:
            res = res + i
        return res

    losses = []

    cls_logits = head_outputs['cls_logits']

    for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        # determine only the foreground
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        num_foreground = foreground_idxs_per_image.sum()

        # create the target classification
        gt_classes_target = torch.zeros_like(cls_logits_per_image).float()
        gt_classes_target[
            foreground_idxs_per_image,

        ] = targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]].float()

        # find indices for which anchors should be ignored
        valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

        # compute the classification loss
        losses.append(sigmoid_focal_loss(
            cls_logits_per_image[valid_idxs_per_image],
            gt_classes_target[valid_idxs_per_image],
            reduction='sum',
        ) / max(1, num_foreground))

    return _sum(losses) / len(targets)

if __name__ == "__main__":
    from src.data.abnormal_dataset import TrainingAbnormalDataSet
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, TrainingVisualizationHook, \
        LogTrainingLoss
    from torch import optim

    from src.training_tasks import BackpropAggregators

    model = RetinaNetFPN()

    dataloader = TrainingAbnormalDataSet()
    dataloader.load_records(keep_annotations=True)

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingAbnormalDataSet)

    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses)
    task.max_iter = 25000

    val_hook = PeriodicStepFuncHook(5000, lambda: task.annotation_validation(val_dl, model))
    checkpoint_hook = ResnetCheckpointHook(1000, "retinanet_test2")

    task.register_hook(LogTrainingLoss())
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(checkpoint_hook)

    task.begin_or_resume()
