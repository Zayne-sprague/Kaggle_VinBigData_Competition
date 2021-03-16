from torch.optim import Adam

from unittest import TestCase

from src.training_tasks.training_task import TrainingTask, SimpleTrainer
from src.training_tasks.tasks.AbnormalClassificationTask import AbnormalClassificationTask

from src.utils.hooks import StepTimer, PeriodicStepFuncHook, CheckpointHook, TrainingVisualizationHook, LogTrainingLoss

from src.models.pseudo.pseudo_model import PseudoModel
from src.models.simple.SimpleNet import SimpleNet
from src.data_loaders.pseudo_dataloader import TrainingPseudoDataloader
from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader


def validation_tester():
    print("Validate")


class TestTrainingTask(TestCase):

    def test_basic(self):
        # model = PseudoModel()
        # dataloader = TrainingPseudoDataloader(num_of_records=100)
        # dataloader.load_records()

        model = SimpleNet()
        dataloader = TrainingAbnormalDataLoader()
        dataloader.load_records(keep_annotations=False)

        train_dl, val_dl = dataloader.partition_data([0.75, 0.25], TrainingAbnormalDataLoader)

        task = AbnormalClassificationTask(model, train_dl, Adam(model.parameters(), lr=0.0001))
        task.max_iter = 100_000_000
        # task = TrainingTask()

        val_hook = PeriodicStepFuncHook(40, lambda: task.validation(val_dl, model))
        checkpoint_hook = CheckpointHook(1, "test")

        task.register_hook(LogTrainingLoss())
        task.register_hook(StepTimer())
        task.register_hook(val_hook)
        # task.register_hook(checkpoint_hook)
        # task.register_hook(TrainingVisualizationHook(batch=False))

        task.begin_or_resume()

        assert 1 == 1

