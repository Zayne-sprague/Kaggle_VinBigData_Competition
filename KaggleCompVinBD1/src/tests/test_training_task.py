from unittest import TestCase

from src.training_tasks.training_task import TrainingTask


class TestTrainingTask(TestCase):

    def test_basic(self):
        task = TrainingTask("test", 100)

        task.begin_or_resume()

        assert 1 == 1

