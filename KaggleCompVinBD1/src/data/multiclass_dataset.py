from tqdm import tqdm
from tabulate import tabulate

from src.data.data_set import TrainingDataSet, TestingDataLoader
from src.utils.cacher import cache
from src import Classifications
from src import config


class TrainingMulticlassDataset(TrainingDataSet):

    def load_records(self):
        records = self.__load_records__()
        self.stats = self.build_metrics(records)
        self.records = records
        self.__annotated__ = True

        return self.records

    @cache(prefix="multiclass_")
    def __load_records__(self):
        records = super().__load_records__()

        num_classes = len(Classifications) if config.include_healthy_annotations else len(Classifications) - 1

        for idx, record in tqdm(enumerate(records), total=len(records)):
            annotations = {'boxes': [], 'labels': [], 'category_ids': []}
            for adx, annotation in enumerate(record['annotations']):
                category_id = annotation['category_id']
                # label = [0] * num_classes
                # label[category_id] = 1
                label = category_id


                annotations['boxes'].append(annotation['bbox'])
                annotations['labels'].append(label)
                annotations['category_ids'].append(category_id)

            records[idx]['annotations'] = annotations

        return records


    def build_metrics(self, records):

        stats = {'total': 0}
        classes = list(Classifications)
        if not config.include_healthy_annotations:
            classes = classes[:-1]

        for cls in classes:
            stats[cls.name] = 0

        for record in records:
            for idx in range(len(record['annotations']['category_ids'])):
                category_id = record['annotations']['category_ids'][idx]
                label = classes[category_id].name

                stats[label] += 1
                stats['total'] += 1

        return stats

    def get_metrics(self) -> dict:
        self.stats = self.build_metrics(self.records)
        return self.stats


    def display_metrics(self, metrics: dict) -> None:
        table = []
        for ky in metrics:
            table.append([ky, metrics[ky]])

        second_row_title = 'Number of Examples'
        if self.__annotated__:
            second_row_title = 'Number of Annotations'

        self.log.info(f'\n-- Abnormal DataSet Metrics --\n{tabulate(table, headers=["Type", second_row_title])}')


class TestingMulticlassDataset(TestingDataLoader):

    def load_records(self):
        self.records = self.__load_records__()
        return self.records

    @cache(prefix="test_multiclass_")
    def __load_records__(self):
        records = super().__load_records__()
        return records


    def build_metrics(self, records):
        stats = {'total': len(records)}
        return stats

    def get_metrics(self) -> dict:
        self.stats = self.build_metrics(self.records)
        return self.stats


    def display_metrics(self, metrics: dict) -> None:
        table = []
        for ky in metrics:
            table.append([ky, metrics[ky]])

        second_row_title = 'Number of Examples'

        self.log.info(f'\n-- Abnormal DataSet Metrics --\n{tabulate(table, headers=["Type", second_row_title])}')
