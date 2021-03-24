import copy
import albumentations as A
import numpy as np

import torch

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

class MyMapper:
    def __init__(self, cfg, is_train: bool = True):
        aug_kwargs = cfg.aug_kwargs
        aug_list = [

        ]
        if is_train:
            aug_list.extend([getattr(T, name)(**kwargs) for name, kwargs in aug_kwargs.items()])

        self.augmentations = T.AugmentationList(aug_list)
        self.is_train = is_train

        mode = 'training' if is_train else 'inference'
        print(f'[MyDatasetMapper] Augmentation used in {mode}: {self.augmentations}')

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict


class AlbumentationsMapper:
    def __init__(self, cfg, is_train: bool = True):
        aug_kwargs = cfg.aug_kwargs
        aug_list = [

        ]
        if is_train:
            aug_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])

        self.transform = A.Compose(aug_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        self.is_train = is_train

        mode = 'training' if is_train else 'inference'
        print(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')

        prev_anno = dataset_dict['annotations']
        bboxes = np.array([obj['bbox'] for obj in prev_anno], dtype=np.float32)
        category_id = np.arange(len(dataset_dict['annotations']))

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_id)
        image = transformed['image']
        annos = []
        for i, j in enumerate(transformed['category_ids']):
            d = prev_anno[j]
            d['bbox'] = transformed['bboxes'][i]
            annos.append(d)
        dataset_dict.pop('annotations', None)

        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict


