from unittest import TestCase
from torch.utils.data import DataLoader

from src.data.abnormal_dataset import TrainingAbnormalDataSet


class TestTrainingAbnormalDataSet(TestCase):
    def test_load_records(self):
        data_loader = TrainingAbnormalDataSet()

        records = data_loader.load_records()

        loader = DataLoader(data_loader, batch_size=4, shuffle=True, num_workers=4)

        for batch in loader:
            print(batch)
            pass

        assert len(records) > 0
