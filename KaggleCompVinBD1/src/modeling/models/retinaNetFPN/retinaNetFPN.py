import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead
from torchvision.ops import sigmoid_focal_loss

from src.modeling.models.model import BaseModel
from src.modeling.models.res50.res50 import Res50
from src.utils.hooks import CheckpointHook
from src.utils.paths import MODELS_DIR

from torchvision.models import resnet50



class RetinaNetFPN(BaseModel):
    def __init__(self):
        super().__init__(model_name="RetinaNetFPN")

        self.m = retinanet_resnet50_fpn(True)

        # Not my favorite code-- but it will get the job done and is nicer than the duck patch.  It also allows
        # native transfer learning from the torchvision package.
        self.m.head = TwoClassRetinaHead(self.m.backbone, self.m.head)

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


class TwoClassRetinaHead(torch.nn.Module):

    def __init__(self, backbone, head: RetinaNetHead):
        super().__init__()

        # TODO - allow transfer learning here, there's only 1 conv layer that cannot be transfered.
        self.classification_head = RetinaNetClassificationHeadOHE(backbone.out_channels, head.classification_head.num_anchors, 2)
        self.regression_head = head.regression_head

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class RetinaNetClassificationHeadOHE(RetinaNetClassificationHead):

    def compute_loss(self, targets, head_outputs, matched_idxs):
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
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, LogTrainingLoss
    from torch import optim

    from src.training_tasks import BackpropAggregators

    model = RetinaNetFPN()

    dataloader = TrainingAbnormalDataSet()
    dataloader.load_records(keep_annotations=True)

    train_dl, val_dl = dataloader.partition_data([0.95, 0.05], TrainingAbnormalDataSet)

    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses)
    task.max_iter = 25000

    val_hook = PeriodicStepFuncHook(20, lambda: task.annotation_validation(val_dl, model))
    checkpoint_hook = CheckpointHook(5, "retinanet_test3")

    task.register_hook(LogTrainingLoss(frequency=1))
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(checkpoint_hook)

    task.begin_or_resume()
