from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog

from tqdm import tqdm

from detectron_modeling.data.detectron_data_set import DetectronTrainingDataSet
from src.utils.cacher import cache
from src import is_record_healthy
from src import CLASSES

class DetectronTrainingMulticlassDataSet(DetectronTrainingDataSet):

    def load_records(self, rad_id=None):
        records = self.__load_records__()
        filtered_records = []

        for i, record in enumerate(records):
            annos = []

            for idx, a in enumerate(record['annotations']):
                if (not rad_id or a['rad_id'] == rad_id) and a['category_id'] != 14:
                    annos.append(a)

            if len(annos) > 0:
                rec = record
                rec['annotations'] = annos
                filtered_records.append(rec)


        self.records = filtered_records
        return self.records

    @cache(prefix="detectron_multiclass_")
    def __load_records__(self):

        records = super().__load_records__()

        for idx, record in tqdm(enumerate(records), total=len(records)):

            for i in range(len(record['annotations'])):
                records[idx]['annotations'][i]['bbox_mode'] = BoxMode.XYXY_ABS

            records[idx]['file_name'] += '.png'

        return records

    def register_metadata(self):
        MetadataCatalog.get(self.name).set(thing_classes=CLASSES[:-1])
