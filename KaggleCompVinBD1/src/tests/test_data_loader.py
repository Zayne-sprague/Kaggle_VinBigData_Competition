from unittest import TestCase

from src.data_loaders.data_loader import TrainingDataLoader


class TestTrainingDataLoader(TestCase):
    def test_load_records(self):
        data_loader = TrainingDataLoader()

        records = data_loader.load_records()

        assert len(records) > 0
