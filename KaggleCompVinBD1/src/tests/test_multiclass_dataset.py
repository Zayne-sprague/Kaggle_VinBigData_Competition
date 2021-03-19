from unittest import TestCase

from src.data.multiclass_dataset import TrainingMulticlassDataset


class TestTrainingMulticlassDataset(TestCase):

    def test_basic(self):
        dataset = TrainingMulticlassDataset()
        records = dataset.load_records()


        assert 1 == 1
