from unittest import TestCase
from torch.utils.data import DataLoader

from src.data_loaders.abnormal_dataloader import TrainingAbnormalDataLoader


class TestTrainingAbnormalDataLoader(TestCase):
    def test_load_records(self):
        data_loader = TrainingAbnormalDataLoader()

        records = data_loader.load_records()

        loader = DataLoader(data_loader, batch_size=4, shuffle=True, num_workers=4)

        for batch in loader:
            print(batch)
            pass

        assert len(records) > 0
