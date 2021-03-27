import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F

import timm

from src.modeling.models.model import BaseModel
from src.modeling.losses.NLLLossOHE import NLLLossOHE
from src.utils.hooks import CheckpointHook
from src import config


class TimmClassifier(BaseModel):
    def __init__(self, timm_model='resnetv2_50x1_bitm', trainable_backbone_layers=3, per_layer_predictions=False):
        super().__init__(model_name=f"TimmModel_{timm_model}")

        self.per_layer_predictions = per_layer_predictions

        self.model = timm.create_model(timm_model, pretrained=True, features_only=True)
        trainable_backbone_layers = min(len(self.model.feature_info.info), trainable_backbone_layers)

        self.num_layers = trainable_backbone_layers

        if per_layer_predictions:
            size = config.image_size
            fcs = []
            for idx, level in enumerate(self.model.feature_info.info[trainable_backbone_layers-1:]):
                lvl_size = size / level['reduction']
                chns = level['num_chs']

                in_features = lvl_size * lvl_size * chns

                fc = nn.Linear(int(in_features), 2)
                # rl = nn.ReLU()
                # fc2 = nn.Linear(1000, 2)


                fcs.append(nn.Sequential(*[fc]))
            self.fcs = nn.ModuleList(fcs)

            for layer in self.fcs.children():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        else:
            in_features = 0
            size = config.image_size
            for idx, level in enumerate(self.model.feature_info.info):
                lvl_size = size / level['reduction']
                chns = level['num_chs']

                in_features += lvl_size * lvl_size * chns


            self.prediction_pipe = nn.Sequential(*[
                nn.Linear(int(in_features), 2)
            ])

            for layer in self.prediction_pipe.children():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        layer_names = list(self.model.return_layers.keys())
        for i in range(max(0, len(self.model.return_layers) - trainable_backbone_layers)):
            layer_name = layer_names[i]
            layer = self.model.__getattr__(layer_name)
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, data: dict) -> dict:
        x = data['image']

        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.float()

        x = x.permute(0, 3, 1, 2)

        # So... this sucks... may have to find a pretrained resnet on grayscale or do it myself which seems unreasonable
        x = x.repeat(1, 3, 1, 1)

        preds = []
        x = self.model(x)

        if self.per_layer_predictions:
            for idx, feature in enumerate(x[self.num_layers - 1:]):
                feature_pred = self.fcs[idx](feature.view([batch_size, -1]))
                preds.append(feature_pred)
            preds = torch.stack(preds)
        else:
            features = [f.view([batch_size, -1]) for f in x]
            features = torch.cat(features, dim=1)
            preds = self.prediction_pipe(features)

        out = {}
        if self.training:
            out['preds'] = preds

            out['losses'] = self.loss(out, data)
        else:
            if self.per_layer_predictions:
                preds = (preds.permute(1, 0, 2).argmax(-1).sum(-1) / self.num_layers > 0.5).int()
            else:
                preds = torch.argmax(preds, -1)
            out = {'preds': preds}



        return out

    def loss(self, predictions: dict, data: dict) -> dict:
        predictions: torch.Tensor = predictions['preds']
        predictions = predictions.view([-1, 2])

        if self.per_layer_predictions:
            labels = data['label'].repeat_interleave(self.num_layers, 0)
        else:
            labels = data['label']

        loss = torch.sum(-labels * F.log_softmax(predictions, -1), -1)
        return {'loss': loss.mean()}



if __name__ == "__main__":
    from src.data.abnormal_dataset import TrainingAbnormalDataSet
    from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask
    from src.utils.hooks import StepTimer, PeriodicStepFuncHook, LogTrainingLoss
    from torch import optim
    from src.data_augs.batch_augmenter import BatchAugmenter
    from src.data_augs.mix_up import MixUpImage
    from src.modeling.lrschedulers import LRScheduler

    from src.training_tasks import BackpropAggregators

    model = TimmClassifier(per_layer_predictions=False)

    dataloader = TrainingAbnormalDataSet()
    dataloader.load_records(keep_annotations=False)

    train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingAbnormalDataSet)

    steps_per_epoch = len(train_dl) // config.artificial_batch_size

    batch_aug = BatchAugmenter()
    batch_aug.compose([MixUpImage(probability=0.75)])
    task = AbnormalClassificationTask(model, train_dl, optim.Adam(model.parameters(), lr=0.0001), backward_agg=BackpropAggregators.MeanLosses, batch_augmenter=batch_aug)

    # task = AbnormalClassificationTask(model, train_dl, optim.SGD(model.parameters(), lr=0.003, momentum=0.9),
    #                                   backward_agg=BackpropAggregators.MeanLosses, batch_augmenter=batch_aug)

    task.max_iter = steps_per_epoch * 25

    val_hook = PeriodicStepFuncHook(steps_per_epoch, lambda: task.tim__validation(val_dl, model))
    train_acc_hook = PeriodicStepFuncHook(steps_per_epoch, lambda: task.tim__validation(train_dl, model))

    checkpoint_hook = CheckpointHook(steps_per_epoch, "TimmModel_BiTRes_X1_TestFive", permanent_checkpoints=steps_per_epoch, keep_last_n_checkpoints=5)

    lr_steps = [1.0, 0.1, 0.01, 0.001]
    # steps = [ steps_per_epoch * 10, steps_per_epoch * 15, steps_per_epoch * 20]
    steps = [ -1, -1, steps_per_epoch * 3]
    def lr_stepper(current_step):
        idx = 0
        for step in steps:
            if current_step <= step:
               return lr_steps[idx]
            idx += 1
        return lr_steps[-1]

    # scheduler = LRScheduler.LinearWarmup(0, steps_per_epoch * 5)
    scheduler2 = LRScheduler.LambdaLR(0, None, lr_stepper)

    task.register_hook(LogTrainingLoss(frequency=20))
    task.register_hook(StepTimer())
    task.register_hook(val_hook)
    task.register_hook(train_acc_hook)
    task.register_hook(checkpoint_hook)

    task.register_lrschedulers(scheduler2)
    # task.register_lrschedulers(scheduler)

    task.log.info(f"Steps per epoch: {steps_per_epoch}")

    task.begin_or_resume()
