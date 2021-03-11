import torch
import random

from src.data_loaders.data_loader import TrainingDataLoader
from src.utils.cacher import cache


class TrainingPseudoDataloader(TrainingDataLoader):

    def __init__(self, num_classes: int = 2, num_of_records: int = 10000):
        super().__init__(readin_annotation_data=False, readin_meta_data=False)

        self.num_classes: int = num_classes
        self.num_of_records: int = num_of_records

    def load_records(self):
        self.records = self.__load_records__()
        return self.records

    def __getitem__(self, idx):
        self.__records_check__()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Data that doesn't have to do large image processing, meant for testing other systems.
        record = self.records[idx]
        record['image'] = 0
        return record

    @cache(prefix="pseudo_")
    def __load_records__(self):
        records = [{'id': x, 'label': random.randint(0, self.num_classes)} for x in range(0, self.num_of_records)]
        self.records = records
        return records


