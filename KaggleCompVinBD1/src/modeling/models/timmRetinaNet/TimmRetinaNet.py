import torch

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.ops import sigmoid_focal_loss
import timm

from collections import OrderedDict

from src.modeling.models.model import BaseModel
from src import config, Classifications, log

class TimmRetinaNet(BaseModel):

    def __init__(self, backbone_timm_model='resnetv2_50x1_bitm'):
        super().__init__("TimmRetinaNet")
        model = timm.create_model(backbone_timm_model, pretrained=True, features_only=True)

        self.m = retinanet_resnet50_fpn(True, trainable_backbone_layers=5)
        self.m.forward_features_model = model
        self.m.backbone.out_channels = 256

        self.m.head = MultiClassRetinaHead(self.m.backbone, self.m.head, [x['num_chs'] for x in model.feature_info.info])

    def forward(self, data: dict) -> dict:
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
                return {'error': True}

            loss = (x['classification'] + x['bbox_regression']).mean()
            out = {'losses': {'loss': loss}, 'other_metrics': {'classifiction_loss': x['classification'].item(),
                                                               'bbox_regression_loss': x['bbox_regression'].item()}}

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
        _features = self.m.forward_features_model(images.tensors)
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

    def __init__(self, backbone, head: RetinaNetHead, channels):
        super().__init__()
        from torch import nn


        # TODO - allow transfer learning here, there's only 1 conv layer that cannot be transfered.
        self.classification_head = RetinaNetClassificationHeadOHE(backbone.out_channels, head.classification_head.num_anchors, len(Classifications) if config.include_healthy_annotations else len(Classifications) - 1)
        self.regression_head = RetinaNetRegressionHead(backbone.out_channels, head.classification_head.num_anchors)


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

if __name__=="__main__":
    from src.utils.hooks import CheckpointHook

    from src.data.multiclass_dataset import TrainingMulticlassDataset
    from src.training_tasks.tasks.MulticlassDetectionTask import MulticlassDetectionTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, LogTrainingLoss
    from torch import optim
    from src.data_augs.batch_augmenter import BatchAugmenter
    from src.data_augs.mix_up import MixUpImageWithAnnotations

    from src.training_tasks import BackpropAggregators

    model = TimmRetinaNet()

    dataloader = TrainingMulticlassDataset()
    dataloader.load_records()

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingMulticlassDataset)

    batch_aug = BatchAugmenter()
    batch_aug.compose([MixUpImageWithAnnotations(probability=0.75)])
    task = MulticlassDetectionTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses, batch_augmenter=batch_aug)
    task.max_iter = 25000

    val_hook = PeriodicStepFuncHook(500, lambda: task.validation(val_dl, model))
    checkpoint_hook = CheckpointHook(250, "retinaNetEnsemble_FullTestTwo", permanent_checkpoints=5000, keep_last_n_checkpoints=5)

    task.register_hook(LogTrainingLoss(frequency=1))
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(checkpoint_hook)

    task.begin_or_resume()
