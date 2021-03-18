from torch.optim import Adam

from unittest import TestCase

from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

from src.utils.hooks import StepTimer, PeriodicStepFuncHook, CheckpointHook, LogTrainingLoss
from src.data_augs.mix_up import MixUpImage

from src.models.simple.SimpleNet import SimpleNet
from src.data.abnormal_dataset import TrainingAbnormalDataSet

from src.data_augs.batch_augmenter import BatchAugmenter
from src.data_augs.mix_up import MixUpImage, MixUpImageWithAnnotations

def validation_tester():
    print("Validate")


class TestTrainingTask(TestCase):

    def test_basic(self):
        # model = PseudoModel()
        # dataloader = PseudoModel()
        # dataloader.load_records()

        model = SimpleNet()
        dataset = TrainingAbnormalDataSet()
        dataset.load_records(keep_annotations=False)


        train_dl, val_dl = dataset.partition_data([0.75, 0.25], TrainingAbnormalDataSet)
        # train_dl.register_augmentation(mix_up_aug)


        batch_aug = BatchAugmenter()
        batch_aug.compose([
            MixUpImage(probability=0.75)
        ])

        task = AbnormalClassificationTask(model, train_dl, Adam(model.parameters(), lr=0.0001), batch_augmenter=batch_aug)
        task.max_iter = 100_000_000
        # task = TrainingTask()

        val_hook = PeriodicStepFuncHook(40, lambda: task.validation(val_dl, model))
        checkpoint_hook = CheckpointHook(1, "test")

        task.register_hook(LogTrainingLoss())
        task.register_hook(StepTimer())
        # task.register_hook(val_hook)
        # task.register_hook(checkpoint_hook)
        # task.register_hook(TrainingVisualizationHook(batch=False))

        task.begin_or_resume()

        assert 1 == 1

