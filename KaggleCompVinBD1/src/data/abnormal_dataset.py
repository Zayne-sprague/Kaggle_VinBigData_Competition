from tqdm import tqdm
from tabulate import tabulate

from src.data.data_set import TrainingDataSet
from src.utils.cacher import cache
from src import is_record_healthy
from src import Classifications

class TrainingAbnormalDataSet(TrainingDataSet):

    def load_records(self, keep_annotations=False):
        if keep_annotations:
            self.__annotated__ = True
            records = self.__load_records_with_annotations__()
        else:
            self.__annotated__ = False
            records = self.__load_records__()

        self.stats = self.build_metrics(records)
        self.records = records

        return self.records

    @cache(prefix="abnormal_")
    def __load_records__(self):
        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['label'] = [1, 0]
            else:
                records[idx]['label'] = [0, 1]

            del records[idx]['annotations']

        return records

    @cache(prefix="abnormal_with_annotations_")
    def __load_records_with_annotations__(self):
        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['annotations'] = {'boxes': [[0, 0, 256, 256]], 'labels': [[1, 0]]}
            else:
                annotations = {'boxes': [], 'labels': []}
                for annotation in records[idx]['annotations']:
                    if annotation['category_id'] != Classifications.Healthy:
                        # TODO - why do we need to explicitly cast to int(), thought the parent class took care of this
                        annotations['boxes'].append([int(x) for x in annotation['bbox']])
                        annotations['labels'].append([0, 1])
                records[idx]['annotations'] = annotations

        return records

    def build_metrics(self, records):

        if not self.__annotated__:
            total = len(records)

            healthy = len([x for x in records if x['label'][0] == 1])

            abnormal = total - healthy
        else:
            total = sum([len(x['annotations']['labels']) for x in records])
            healthy = sum([len(x['annotations']['labels']) for x in records if x['annotations']['labels'][0][0] == 1])
            abnormal = total - healthy

        return {
            'total': total,
            'healthy': healthy,
            'abnormal': abnormal
        }

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

