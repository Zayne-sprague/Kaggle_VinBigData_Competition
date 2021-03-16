import torch
import random
from tabulate import tabulate

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
        record['label'] = 0
        return record

    def process_data(self, data):
        for x in data:
            x['image'] = 0
            x['label'] = 0
            yield x

    @cache(prefix="pseudo_")
    def __load_records__(self):
        records = [{'id': x, 'label': random.randint(0, self.num_classes)} for x in range(0, self.num_of_records)]
        self.records = records
        return records

    def get_metrics(self) -> dict:
        self.__records_check__()

        total = len(self.records)

        return {
            'total': total,
        }

    def display_metrics(self, metrics: dict) -> None:
        table = []
        for ky in metrics:
            table.append([ky, metrics[ky]])

        self.log.info(f'\n-- Pseudo Dataloader Metrics --\n{tabulate(table, headers=["Type", "Number Of Examples"])}')


