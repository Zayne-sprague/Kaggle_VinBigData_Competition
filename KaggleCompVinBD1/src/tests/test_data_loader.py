from unittest import TestCase

from src.data.data_set import TrainingDataSet


class TestTrainingDataSet(TestCase):
    def test_load_records(self):
        data_set = TrainingDataSet()

        records = data_set.load_records()

        assert len(records) > 0
