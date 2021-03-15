from torch.optim import Adam

from unittest import TestCase

from src.training_tasks.training_task import TrainingTask, SimpleTrainer

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

        task = SimpleTrainer(model, dataloader, Adam(model.parameters(), lr=0.0001))
        task.max_iter = 100_000_000
        # task = TrainingTask()

        val_hook = PeriodicStepFuncHook(1, validation_tester)
        checkpoint_hook = CheckpointHook("checkpoint_test", 1)

        task.register_hook(StepTimer())
        # task.register_hook(val_hook)
        # task.register_hook(checkpoint_hook)
        task.register_hook(TrainingVisualizationHook(batch=False))
        task.register_hook(LogTrainingLoss())
        task.begin_or_resume()

        assert 1 == 1

