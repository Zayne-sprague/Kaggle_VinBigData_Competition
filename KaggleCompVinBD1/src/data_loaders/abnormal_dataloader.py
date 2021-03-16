from tqdm import tqdm
from tabulate import tabulate

from src.data_loaders.data_loader import TrainingDataLoader
from src.utils.cacher import cache
from src import is_record_healthy


class TrainingAbnormalDataLoader(TrainingDataLoader):

    def load_records(self, keep_annotations=False):
        self.records = self.__load_records__(keep_annotations)
        return self.records

    @cache(prefix="abnormal_")
    def __load_records__(self, keep_annotations: bool = False):

        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['label'] = 0
            else:
                records[idx]['label'] = 1

            if not keep_annotations:
                del records[idx]['annotations']

        return records

    def get_metrics(self) -> dict:
        self.__records_check__()

        total = len(self.records)
        healthy = len([x for x in self.records if x['label'] == 0])
        abnormal = total - healthy

        return {
            'total': total,
            'healthy': healthy,
            'abnormal': abnormal
        }

    def display_metrics(self, metrics: dict) -> None:
        table = []
        for ky in metrics:
            table.append([ky, metrics[ky]])

        self.log.info(f'\n-- Abnormal Dataloader Metrics --\n{tabulate(table, headers=["Type", "Number Of Examples"])}')

