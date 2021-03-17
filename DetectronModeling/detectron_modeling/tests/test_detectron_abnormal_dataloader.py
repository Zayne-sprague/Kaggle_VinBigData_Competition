import random
import cv2

from unittest import TestCase
from torch.utils.data import DataLoader

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron_modeling.data_loaders.detectron_abnormal_dataset import DetectronTrainingAbnormalDataSet


class TestDetectronTrainingAbnormalDataSet(TestCase):
    def test_load_records(self):
        data_set = DetectronTrainingAbnormalDataSet("training_data")

        records = data_set.load_records()
        data_set.register_records()
        data_set.register_metadata()

        data = DatasetCatalog.get("training_data")
        metadata = MetadataCatalog.get("training_data")

        for d in random.sample(data, 10):
            img = cv2.imread(d['file_name'])
            visualizer = Visualizer(img[:, :,::-1], metadata=metadata, scale=1)

            vis = visualizer.draw_dataset_dict(d)

            cv2.imshow('image', vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        assert len(records) > 0
