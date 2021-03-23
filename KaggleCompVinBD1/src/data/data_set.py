from torch.utils.data import IterableDataset, random_split, Subset, Dataset
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import weakref


import cv2
from skimage import io
from tqdm import tqdm
import logging
from itertools import cycle

from src.utils.paths import TRAINING_ANNOTATION_DATA, TRAIN_META_DATA, CONVERTED_VBD_DICOM_DATA_FOLDER, TEST_META_DATA
from src.utils.cacher import cache
from src import config, dl_log, Classifications

__NUM_OF_IMGS_WITH_DEBUG__ = 1250
__NUM_OF_TEST_IMGS_WITH_DEBUG__ = 25


#https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
# TODO implement parallel data loading and re-read ^ (also shuffling would be nice)
class TrainingDataSet:
    # Base class for data loaders

    def __init__(self, readin_annotation_data=True, readin_meta_data=True, image_transformations=None):
        # If you are not reading in the annotation/meta data, you better set it yourself (usually you only wouldn't want
        # to read in this data if you are testing the implementation of other systems with this class-- i.e. speed)
        if readin_annotation_data:
            self.annotation_data: pd.DataFrame = pd.read_csv(TRAINING_ANNOTATION_DATA)
        else:
            self.annotation_data = None

        if readin_meta_data:
            self.meta_data: pd.DataFrame = pd.read_csv(TRAIN_META_DATA)
        else:
            self.meta_data = None

        self.image_transformations = image_transformations

        self.image_size: int = config.image_size
        self.__annotated__: bool = False

        self.image_dir: Path = CONVERTED_VBD_DICOM_DATA_FOLDER / f'{self.image_size}' / 'train'

        self.log: logging.Logger = dl_log

        self.debug: bool = config.DEBUG

        if self.debug and readin_meta_data:
            self.meta_data = self.meta_data.iloc[:__NUM_OF_IMGS_WITH_DEBUG__]

        self.records = None

        self.stats = {}

    def __records_check__(self):
        if not self.records:
            self.log.error("Attempted to access data loaders records without loading them first!")
            raise Exception("Load records for dataloader before accessing them")


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
                rad_id = annotation['rad_id']

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
                    "category_id": class_id,
                    "rad_id": rad_id
                })

            if not config.include_records_without_annotations and len(annotations) == 0:
                continue

            record['annotations'] = annotations
            records.append(record)

        self.log.info("Finished loading records")

        self.records = records
        return records

    #TODO - fix the type pass in... maybe let the callee handle it after all?
    def partition_data(self, partitions: List[float], type):
        self.__records_check__()
        assert all([0.0 <= x <= 1.0 for x in partitions]), "Please pass in values for the partitions between 0 and 1"
        assert sum(partitions) >= 1.0 - .005  # epsilon

        partition_counts = [int(x * len(self.records)) for x in partitions]
        partition_counts[-1] += int(len(self.records) - sum(partition_counts))

        parts = random_split(self, partition_counts)
        for part in parts:
            dl = type(readin_annotation_data=False, readin_meta_data=False)
            dl.records = np.array(self.records)[part.indices].tolist()
            dl.meta_data = self.meta_data
            dl.annotation_data = self.annotation_data
            dl.__annotated__ = self.__annotated__
            # Really we could be returning them all at once-- but just incase we ever want to make this parallel
            # it makes since to just yield and do list(partition_data([0.25, 0.75])) if you want it all in one go
            # You can also just do a, b, c, d = partition_data([0.25, 0.25, 0.25, 0.25]) which is cool!
            yield dl

    def query(self, query: str) -> pd.DataFrame:
        if self.annotation_data is None:
            self.log.warn("Attempted to query annotation data without annotation file loaded")
            return pd.DataFrame([])
        return self.annotation_data.query(query)

    def m_query(self, query: str) -> pd.DataFrame:
        if self.meta_data is None:
            self.log.warn("Attempted to query meta data without the meta data file loaded")
            return pd.DataFrame([])
        return self.meta_data.query(query)

    def get_metrics(self) -> dict:
        raise NotImplementedError()

    def display_metrics(self, metrics: dict) -> None:
        raise NotImplementedError()


class TrainingDatasetMixin(Dataset):

    def __init__(self, dataset: TrainingDataSet):
        self.dataset = dataset
        self.log = self.dataset.log

    @property
    def records(self):
        return self.dataset.records

    def __len__(self):
        self.dataset.__records_check__()
        return len(self.dataset.records)

    def __getitem__(self, idx):
        self.dataset.__records_check__()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.records[idx]

        if 'label' in record:
            record['label'] = np.array(record['label'])
        if 'annotations' in record and 'label' not in record:
            record['annotations'] = {'boxes': np.array(record['annotations']['boxes']), 'labels': np.array(record['annotations']['labels'])}

        if 'file_name' in record:
            image = io.imread(record['file_name'] + '.png')

            if self.dataset.image_transformations:
                image = self.dataset.image_transformations(image=image)['image']

            record['image'] = image
        else:
            record['image'] = 0

        return record


class TrainingDatasetIterable(IterableDataset):

    def __init__(self, dataset: TrainingDataSet):
        self.dataset = dataset
        self.log = self.dataset.log

    def __len__(self):
        self.dataset.__records_check__()
        return len(self.dataset.records)

    def __getitem__(self, idx):
        self.dataset.__records_check__()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.dataset.records[idx]

        if 'label' in record:
            record['label'] = np.array(record['label'])
        if 'annotations' in record:
            record['annotations'] = {'boxes': np.array(record['annotations']['boxes']), 'labels': np.array(record['annotations']['labels'])}

        if 'file_name' in record:
            image = io.imread(record['file_name'] + '.png')

            if self.dataset.image_transformations:
                image = self.dataset.image_transformations(image=image)['image']

            record['image'] = image
        else:
            record['image'] = 0

        return record

    @property
    def records(self):
        return self.dataset.records

    def process_data(self, data):
        for x in data:
            if 'file_name' in x:
                x['image'] = io.imread(x['file_name'] + '.png')

            yield x

    def __get_stream__(self, data):
        return cycle(self.process_data(data))

    def __iter__(self):
        self.dataset.__records_check__()

        return self.__get_stream__(self.dataset.records)


class TestingDataLoader(TrainingDataSet):

    # Base class for data loaders

    def __init__(self, readin_meta_data=True):
        super().__init__(readin_annotation_data=False, readin_meta_data=False)

        if readin_meta_data:
            self.meta_data: pd.DataFrame = pd.read_csv(TEST_META_DATA)
        else:
            self.meta_data = None

        if self.debug and readin_meta_data:
            self.meta_data = self.meta_data.iloc[:__NUM_OF_TEST_IMGS_WITH_DEBUG__]

        self.__annotated__ = True
        self.image_dir: Path = CONVERTED_VBD_DICOM_DATA_FOLDER / f'{self.image_size}' / 'test'

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

        if 'filename' in record:
            image = io.imread(record['file_name'] + '.png')

            record['image'] = image
        else:
            record['image'] = 0

        return record

    def process_data(self, data):
        for x in data:
            if 'file_name' in x:
                x['image'] = io.imread(x['file_name'] + '.png')

            yield x

    def __get_stream__(self, data):
        return cycle(self.process_data(data))

    def __iter__(self):
        self.__records_check__()

        return self.__get_stream__(self.records)

    def load_records(self):
        self.records = self.__load_records__()
        return self.records

    @cache(prefix="test_base_")
    def __load_records__(self):

        self.log.info("Loading records")

        records = []

        for index, meta_data_row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data)):
            image_id, im_height, im_width = meta_data_row.values

            record = {
                'file_name': str(self.image_dir / f'{image_id}'),
                'image_id': image_id,
                'height': self.image_size,
                'width': self.image_size,
                'orig_h': im_height,
                'orig_w': im_width
            }

            records.append(record)

        self.log.info("Finished loading records")

        self.records = records
        return records




def get_img_dims(
        imgpath: Path
):
    image = cv2.imread(str(imgpath))
    h, w, c = image.shape

    return h, w, c

