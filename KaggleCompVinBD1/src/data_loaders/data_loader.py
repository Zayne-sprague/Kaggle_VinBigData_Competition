from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path

import cv2
from skimage import io
from tqdm import tqdm
import logging

from src.utils.paths import TRAINING_ANNOTATION_DATA, TRAIN_META_DATA, CONVERTED_VBD_DICOM_DATA_FOLDER
from src.utils.cacher import cache
from src import config, dl_log, Classifications

__NUM_OF_IMGS_WITH_DEBUG__ = 1250


class TrainingDataLoader(Dataset):
    # Base class for data loaders

    def __init__(self, readin_annotation_data=True, readin_meta_data=True):
        # If you are not reading in the annotation/meta data, you better set it yourself (usually you only wouldn't want
        # to read in this data if you are testing the implementation of other systems with this class-- i.e. speed)
        if readin_annotation_data:
            self.annotation_data: pd.DataFrame = pd.read_csv(TRAINING_ANNOTATION_DATA)

        if readin_meta_data:
            self.meta_data: pd.DataFrame = pd.read_csv(TRAIN_META_DATA)

        self.image_size: int = config.image_size

        self.image_dir: Path = CONVERTED_VBD_DICOM_DATA_FOLDER / f'{self.image_size}' / 'train'

        self.log: logging.Logger = dl_log

        self.debug: bool = config.DEBUG

        if self.debug and readin_meta_data:
            self.meta_data = self.meta_data.iloc[:__NUM_OF_IMGS_WITH_DEBUG__]

        self.records = None

    def __records_check__(self):
        if not self.records:
            self.log.error("Attempted to access data loaders records without loading them first!")
            raise Exception("Load records for dataloader before accessing them")

    def __len__(self):
        self.__records_check__()
        return len(self.records)

    def __getitem__(self, idx):
        self.__records_check__()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.records[idx]

        image = io.imread(record['file_name'] + '.png')

        record['image'] = image

        return record

    def load_records(self):
        self.records = self.__load_records__()
        return self.records

    @cache(prefix="base_")
    def __load_records__(self):

        self.log.info("Loading records")

        records = []

        for index, meta_data_row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data)):
            image_id, im_height, im_width = meta_data_row.values

            record = {
                'file_name': str(self.image_dir / f'{image_id}'),
                'image_id': index,
                'height': self.image_size,
                'width': self.image_size
            }

            annotations = []
            for annotation_idx, annotation in self.annotation_data.query("image_id == @image_id").iterrows():
                class_id = annotation['class_id']

                if not config.include_healthy_annotations and class_id == Classifications.Healthy.value:
                    continue

                if class_id == Classifications.Healthy.value:
                    # This may not be the correct way to indicate healthy bounding boxes
                    bbox = [0, 0, self.image_size, self.image_size]
                else:
                    h_ratio = self.image_size / im_height
                    w_ratio = self.image_size / im_width

                    bbox = [
                        int(annotation['x_min'] * w_ratio),
                        int(annotation['y_min'] * h_ratio),
                        int(annotation['x_max'] * w_ratio),
                        int(annotation['y_max'] * h_ratio)
                    ]

                annotations.append({
                    "bbox": bbox,
                    "category_id": class_id
                })

            if not config.include_records_without_annotations and len(annotations) == 0:
                continue

            record['annotations'] = annotations
            records.append(record)

        self.log.info("Finished loading records")

        self.records = records
        return records


class TestingDataLoader:
    pass


def get_img_dims(
        imgpath: Path
):
    image = cv2.imread(str(imgpath))
    h, w, c = image.shape

    return h, w, c