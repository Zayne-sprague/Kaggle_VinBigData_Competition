from tqdm import tqdm

from src.data_loaders.data_loader import TrainingDataLoader
from src.utils.cacher import cache
from src import is_record_healthy


class TrainingAbnormalDataLoader(TrainingDataLoader):

    def __init__(self):
        super().__init__()

    def load_records(self, keep_records=False):
        self.records = self.__load_records__(keep_records)
        return self.records

    @cache(prefix="abnormal_")
    def __load_records__(self, keep_records: bool = False):

        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['label'] = 0
            else:
                records[idx]['label'] = 1

            if not keep_records:
                del records[idx]['annotations']

        return records


