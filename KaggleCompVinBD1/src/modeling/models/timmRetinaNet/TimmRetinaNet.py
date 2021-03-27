import torch
from torch import nn

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.retinanet import det_utils
from torchvision.ops import sigmoid_focal_loss
import timm

import numpy as np
from collections import OrderedDict
import math

from src.modeling.models.model import BaseModel
from src.modeling.models.timmClassifier.timmClassifier import TimmClassifier
from src import config, Classifications, log

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class TimmRetinaNet(BaseModel):

    def __init__(self, load_model=None, backbone_timm_model='resnetv2_50x1_bitm', backbone_channel_size=32, trainable_backbone_layers=3, raise_errors=False, convs_for_head: int = 3, pretrain_retina_net=False, half=True):
        super().__init__("TimmRetinaNet")

        self.m = retinanet_resnet50_fpn(pretrain_retina_net, trainable_backbone_layers=trainable_backbone_layers)

        if load_model:
            model = TimmClassifier()
            model.load(load_model)
            model = model.model
            self.m.backbone = model
            self.m.backbone.out_channels = backbone_channel_size

        else:
            model = timm.create_model(backbone_timm_model, pretrained=True, features_only=True)

            self.m.backbone = model
            self.m.backbone.out_channels = backbone_channel_size

        self.m.head = MultiClassRetinaHead(self.m.backbone, self.m.head, [x['num_chs'] for x in model.feature_info.info], convs_for_head=convs_for_head)

        self.raise_errors = raise_errors

        layer_names = list(model.return_layers.keys())
        for i in range(max(0, len(model.return_layers) - trainable_backbone_layers)):
            layer_name = layer_names[i]
            layer = self.m.backbone.__getattr__(layer_name)
            for param in layer.parameters():
                param.requires_grad = False

        torch.cuda.empty_cache()

        self.__half = half

    def forward(self, data: dict) -> dict:
        if self.__half:
            x = data['image'].half()
        else:
            x = data['image']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        if self.training:
            targets = data['annotations']

            try:
                x = self.retina_forward(x, targets)
            except Exception as e:
                self.log.critical(f"ERROR in model forward: {e}")
                if self.raise_errors:
                    raise e

                return {'error': True}

            loss = (x['classification'] + x['bbox_regression']).mean()
            out = {'losses': {'loss': loss}, 'other_metrics': {'classifiction_loss': x['classification'],
                                                               'bbox_regression_loss': x['bbox_regression']}}

            return out
        else:
            x = self.retina_forward(x)

            return {'preds': x}

    def retina_forward(self, images, targets=None):
        if self.m.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.m.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.m.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        _features = self.m.backbone(images.tensors)
        features = {}
        for idx, feature in enumerate(_features):
            features[idx] = feature
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.m.head(features)

        # create the set of anchors
        anchors = self.m.anchor_generator(images, features)

        losses = {}
        detections = []
        if self.m.training:
            assert targets is not None

            # compute the losses
            losses = self.m.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.m.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.m.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                self.log.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self.m._has_warned = True
            return losses, detections
        return self.m.eager_outputs(losses, detections)

class MultiClassRetinaHead(torch.nn.Module):

    def __init__(self, backbone, head: RetinaNetHead, channels, convs_for_head: int = 3):
        super().__init__()

        self.classification_head = RetinaNetClassificationHeadOHE(backbone.out_channels, head.classification_head.num_anchors, len(Classifications) if config.include_healthy_annotations else len(Classifications) - 1, convs_for_head=convs_for_head)
        self.regression_head = CustomRetinaNetRegressionHead(backbone.out_channels, head.classification_head.num_anchors, convs_for_head=convs_for_head)


        self.scalings = []
        for in_chan in channels:
            scaled_conv = nn.Sequential(*[
                nn.Conv2d(in_chan, backbone.out_channels, kernel_size=1, stride=1),
                nn.ReLU()
            ])

            for layer in scaled_conv.children():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

            self.scalings.append(scaled_conv)

        self.scalings = nn.ModuleList(self.scalings)

        torch.cuda.empty_cache()


    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):

        x = [self.scalings[idx](x[idx]) for idx in range(len(x))]

        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class RetinaNetClassificationHeadOHE(nn.Module):

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, convs_for_head=3):
        super().__init__()

        conv = []
        for _ in range(convs_for_head):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

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

    def forward(self, x):
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class CustomRetinaNetRegressionHead(RetinaNetRegressionHead):

    def __init__(self, in_channels, num_anchors, convs_for_head: int = 3):
        super().__init__(in_channels, num_anchors)

        conv = []
        for _ in range(convs_for_head):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)


if __name__=="__main__":
    from src.utils.hooks import CheckpointHook

    from src.data.multiclass_dataset import TrainingMulticlassDataset
    from src.training_tasks.tasks.MulticlassDetectionTask import MulticlassDetectionTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, LogTrainingLoss
    from torch import optim
    from src.data_augs.batch_augmenter import BatchAugmenter
    from src.data_augs.mix_up import MixUpImageWithAnnotations

    from src.training_tasks import BackpropAggregators
    from src.modeling.lrschedulers import LRScheduler

    HALF = False

    # Not a random number of channels... this is borderline as much memory as I can run with current setup
    # TODO - find ways of optimizing memory so we can increase model size as well as channels per backbone layer
    model = TimmRetinaNet(load_model='TimmModel_BiTRes_X1_TestFive@14060', backbone_channel_size=200, trainable_backbone_layers=5, raise_errors=False, convs_for_head=1, half=HALF)
    if HALF:
        model = model.half()

    dataloader = TrainingMulticlassDataset()
    dataloader.load_records()

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingMulticlassDataset)

    steps_per_epoch = len(train_dl) // config.artificial_batch_size


    batch_aug = BatchAugmenter()
    batch_aug.compose([MixUpImageWithAnnotations(probability=0.5)])
    task = MulticlassDetectionTask(model, train_dl, optim.Adam(model.parameters(), lr=0.001), backward_agg=BackpropAggregators.MeanLosses, batch_augmenter=batch_aug)

    # Loss exploded with this optimizer
    # task = MulticlassDetectionTask(model, train_dl, optim.SGD(model.parameters(), lr=0.003, momentum=0.9), backward_agg=BackpropAggregators.MeanLosses, batch_augmenter=batch_aug)

    task.max_iter = steps_per_epoch * 2500

    validation_iteration = 500
    train_acc_hook = PeriodicStepFuncHook(validation_iteration * 6, lambda: task.validation(train_dl, model))
    val_hook = PeriodicStepFuncHook(validation_iteration, lambda: task.validation(val_dl, model))

    checkpoint_hook = CheckpointHook(validation_iteration, "timmResNetTest14_pretrained_X1", permanent_checkpoints=validation_iteration, keep_last_n_checkpoints=0)

    lr_steps = [1.0, 0.1, 0.01, 0.001, 0.0001]
    steps = [ steps_per_epoch * 5, steps_per_epoch * 10, steps_per_epoch * 15]

    def lr_stepper(current_step):
        idx = 0
        for step in steps:
            if current_step <= step:
               return lr_steps[idx]
            idx += 1
        return lr_steps[-1]


    scheduler = LRScheduler.LinearWarmup(0, steps_per_epoch * 2)
    scheduler2 = LRScheduler.LambdaLR(0, None, lr_stepper)

    task.register_hook(LogTrainingLoss(frequency=20))
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(train_acc_hook)
    task.register_hook(checkpoint_hook)

    task.register_lrschedulers(scheduler2)
    task.register_lrschedulers(scheduler)

    task.log.info(f'Steps per epoch {steps_per_epoch}')
    task.begin_or_resume()
