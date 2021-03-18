from torch.optim import Adam

from unittest import TestCase

from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

from src.utils.hooks import StepTimer, PeriodicStepFuncHook, CheckpointHook, LogTrainingLoss

from src.models.simple.SimpleNet import SimpleNet
from src.models.pseudo.pseudo_model import PseudoModel
from src.data.abnormal_dataset import TrainingAbnormalDataSet
from src.data.pseudo_dataset import TrainingPseudoDataSet

from src.data_augs.batch_augmenter import BatchAugmenter
from src.data_augs.mix_up import MixUpImage, MixUpImageWithAnnotations

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

        task = AbnormalClassificationTask(model, train_dl, Adam(model.parameters(), lr=0.0001), batch_augmenter=batch_aug)
        task.max_iter = 100_000_000
        # task = TrainingTask()

        val_hook = PeriodicStepFuncHook(40, lambda: task.validation(val_dl, model))
        checkpoint_hook = CheckpointHook(1000, "test", 10000, 5)

        task.register_hook(LogTrainingLoss())
        task.register_hook(StepTimer())
        # task.register_hook(val_hook)
        task.register_hook(checkpoint_hook)
        # task.register_hook(TrainingVisualizationHook(batch=False))

        task.begin_or_resume()

        assert 1 == 1

