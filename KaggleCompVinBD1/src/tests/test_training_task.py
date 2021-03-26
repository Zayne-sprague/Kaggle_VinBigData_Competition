from torch.optim import Adam, SGD

from unittest import TestCase

from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

from src.utils.hooks import StepTimer, PeriodicStepFuncHook, CheckpointHook, LogTrainingLoss

from src.modeling.models.pseudo.pseudo_model import PseudoModel
from src.modeling.lrschedulers import LRScheduler
from src.data.pseudo_dataset import TrainingPseudoDataSet

from src.data_augs.batch_augmenter import BatchAugmenter


def validation_tester():
    print("Validate")


class TestTrainingTask(TestCase):

    def test_basic(self):
        model = PseudoModel()
        dataset = TrainingPseudoDataSet()
        dataset.load_records()
        train_dl, val_dl = dataset.partition_data([0.75, 0.25], TrainingPseudoDataSet)


        # model = SimpleNet()
        # dataset = TrainingAbnormalDataSet()
        # dataset.load_records(keep_annotations=True)
        # train_dl, val_dl = dataset.partition_data([0.75, 0.25], TrainingAbnormalDataSet)


        batch_aug = BatchAugmenter()
        # batch_aug.compose([
        #     MixUpImageWithAnnotations(probability=1.0)
        # ])

        task = AbnormalClassificationTask(model, train_dl, SGD(model.parameters(), lr=0.03, momentum=0.9), batch_augmenter=batch_aug)
        # task = AbnormalClassificationTask(model, train_dl, Adam(model.parameters(), lr=0.03, betas=(0.9, 0.999), weight_decay=0.01), batch_augmenter=batch_aug)

        task.max_iter = 100_000_000
        # task = TrainingTask()

        val_hook = PeriodicStepFuncHook(400000, lambda: task.validation(val_dl, model))
        checkpoint_hook = CheckpointHook(100000, "test", 1000000, 5)

        scheduler = LRScheduler.LinearWarmup(0, 3000)
        scheduler2 = LRScheduler.LambdaLR(0, 3000, lambda step: 1.0)

        task.register_hook(LogTrainingLoss(frequency=100))
        task.register_hook(StepTimer())
        # task.register_hook(val_hook)
        task.register_hook(checkpoint_hook)
        # task.register_hook(TrainingVisualizationHook(batch=False))

        task.register_lrschedulers(scheduler2)
        task.register_lrschedulers(scheduler)

        task.begin_or_resume()

        assert 1 == 1

