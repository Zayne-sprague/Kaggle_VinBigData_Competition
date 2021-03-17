from tqdm import tqdm
from tabulate import tabulate

from src.data.data_set import TrainingDataSet
from src.utils.cacher import cache
from src import is_record_healthy


class TrainingAbnormalDataSet(TrainingDataSet):

    def load_records(self, keep_annotations=False):
        records = self.__load_records__(keep_annotations)

        self.stats = self.build_metrics(records)
        self.records = records

        return self.records

    @cache(prefix="abnormal_")
    def __load_records__(self, keep_annotations: bool = False):

        records = super().__load_records__()

        healthy = 0
        abnormal = 0
        total = 0

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['label'] = [1, 0]
                healthy += 1
            else:
                records[idx]['label'] = [0, 1]
                abnormal += 1
            total += 1

            if not keep_annotations:
                del records[idx]['annotations']

        return records

    def build_metrics(self, records):

        total = len(records)

        healthy = len([x for x in records if x['label'][0] == 1])

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

        self.log.info(f'\n-- Abnormal DataSet Metrics --\n{tabulate(table, headers=["Type", "Number Of Examples"])}')

