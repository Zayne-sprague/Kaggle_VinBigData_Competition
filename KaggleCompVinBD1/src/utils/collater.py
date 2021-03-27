import torch

from typing import Optional

from src.data_augs.batch_augmenter import BatchAugmenter
from src import config

class Collater:

    def __init__(self, batch_augmenter: Optional[BatchAugmenter] = None):
        self.batch_augmenter = batch_augmenter

    def __call__(self, batch):
        return self.handle_batch(batch)

    def handle_batch(self, batch):
        raise NotImplementedError()


class SimpleCollater(Collater):

    def handle_batch(self, batch):
        image = torch.tensor([item['image'] for item in batch])

        if 'label' in batch[0]:
            label = torch.tensor([item['label'] for item in batch], dtype=torch.float)

            batch = {'image': image, 'label': label}
        elif 'annotations' in batch[0]:
            annotations = [{
                'boxes': torch.tensor(x['annotations']['boxes']).to(config.devices[1]),
                'labels': torch.tensor(x['annotations']['labels'], dtype=torch.float).to(config.devices[1]),
                'category_ids': [] if 'category_ids' not in x['annotations'] else torch.tensor(x['annotations']['category_ids']).to(config.devices[1]),
            } for x in batch]

            batch = {'image': image, 'annotations': annotations}

        if self.batch_augmenter:
            batch = self.batch_augmenter.transform(batch)

        return batch