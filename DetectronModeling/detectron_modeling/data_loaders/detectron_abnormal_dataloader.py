from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog

from tqdm import tqdm

from detectron_modeling.data_loaders.detectron_data_loader import DetectronTrainingDataLoader
from src.utils.cacher import cache
from src import is_record_healthy


class DetectronTrainingAbnormalDataLoader(DetectronTrainingDataLoader):

    def load_records(self):
        self.records = self.__load_records__()
        return self.records

    @cache(prefix="detectron_abnormal_")
    def __load_records__(self):

        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):
            if is_record_healthy(record):
                records[idx]['annotations'][0]['category_id'] = 0
                records[idx]['annotations'][0]['bbox_mode'] = BoxMode.XYXY_ABS

                records[idx]['annotations'] = [records[idx]['annotations'][0]]

            else:
                for i in range(len(record['annotations'])):
                    records[idx]['annotations'][i]['category_id'] = 1
                    records[idx]['annotations'][i]['bbox_mode'] = BoxMode.XYXY_ABS

            records[idx]['file_name'] += '.png'

        return records

    def register_metadata(self):
        MetadataCatalog.get(self.name).set(thing_classes=[
            "Healthy",
            "Abnormal"
        ])
