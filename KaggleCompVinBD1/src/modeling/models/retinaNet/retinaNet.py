import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead
from torchvision.ops import sigmoid_focal_loss

from collections import OrderedDict


from src.modeling.models.model import BaseModel
from src.modeling.models.res50.res50 import Res50
from src.utils.hooks import CheckpointHook
from src.utils.paths import MODELS_DIR
from src.modeling.losses.NLLLossOHE import NLLLossOHE

import math



class RetinaNet(BaseModel):
    def __init__(self):
        super().__init__(model_name="RetinaNet")

        self.m = retinanet_resnet50_fpn(True, trainable_backbone_layers=5)

        # Not my favorite code-- but it will get the job done and is nicer than the duck patch.  It also allows
        # native transfer learning from the torchvision package.
        self.m.head = WholeImageRetinaHead(self.m.backbone, self.m.head)

        self.criterion = NLLLossOHE()

    def forward(self, data: dict) -> dict:
        x = data['image']
        labels = data['label']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        # TODO - we need to remove this to allow for mixup
        # for idx, annotation in enumerate(annotations):
        #     annotations[idx]['labels'] = torch.argmax(annotation['labels'], 1)
        # x = self.m(x)
        x = self.retina_forward(x, labels)

        if self.training:
            loss = x['classification']
            out = {'losses': {'loss': loss}}

            return out
        else:
            return {'preds': x}

    def retina_forward(self, images, labels):
        # get the original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.m.transform(images, None)

        # Check for degenerate boxes

        # get the features from the backbone
        features = self.m.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.m.head(features)

        losses = {}
        detections = []
        if self.m.training:
            # compute the losses
            losses = self.m.head.compute_loss(labels, head_outputs)
        else:
            detections = head_outputs['cls_logits'].exp()

        if torch.jit.is_scripting():
            if not self.m._has_warned:
                self.m._has_warned = True
            return losses, detections
        return self.m.eager_outputs(losses, detections)

    def loss(self, predictions: dict, data: dict) -> dict:
        predictions: torch.Tensor = predictions['preds']
        loss = self.criterion(predictions, data['label'])
        return {'loss': loss}

class WholeImageRetinaHead(torch.nn.Module):

    def __init__(self, backbone, head: RetinaNetHead):
        super().__init__()

        # TODO - allow transfer learning here, there's only 1 conv layer that cannot be transfered.
        self.classification_head = WholeImageRetinaNetClassificationHead(backbone.out_channels, 2)

    def compute_loss(self, targets, head_outputs):
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs),
        }

    def forward(self, x):
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': torch.tensor([])
        }

class WholeImageRetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_classes, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.fc1 = nn.Linear(26686, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.lsoft = nn.LogSoftmax(dim=-1)

        self.criterion = NLLLossOHE()

        self.num_classes = num_classes

    def compute_loss(self, targets, head_outputs):

        cls_logits = head_outputs['cls_logits']
        loss = NLLLossOHE()(cls_logits, targets)

        return loss

    def forward(self, x):
        all_cls_logits = []

        batch_size = x[0].shape[0]

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        x = torch.cat(all_cls_logits, dim=1)
        x = x.view([batch_size, -1])
        x = self.fc1(x)
        x = self.fc2(x)
        predictions = self.lsoft(x)

        return predictions




if __name__ == "__main__":
    from src.data.abnormal_dataset import TrainingAbnormalDataSet
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, LogTrainingLoss
    from torch import optim

    from src.training_tasks import BackpropAggregators

    model = RetinaNet()

    dataloader = TrainingAbnormalDataSet()
    dataloader.load_records(keep_annotations=False)

    train_dl, val_dl = dataloader.partition_data([0.95, 0.05], TrainingAbnormalDataSet)

    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses)
    task.max_iter = 25000

    val_hook = PeriodicStepFuncHook(1, lambda: task.validation(val_dl, model))
    checkpoint_hook = CheckpointHook(5, "retinanet_backbone_test")

    task.register_hook(LogTrainingLoss(frequency=1))
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(checkpoint_hook)

    task.begin_or_resume()
