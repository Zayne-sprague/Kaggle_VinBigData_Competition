import torch

import numpy as np
import random
from typing import Tuple

from .batch_augmenter import Augmentor



# https://arxiv.org/pdf/1710.09412.pdf
class MixUpImage(Augmentor):

    def handle_transform(self, batch: dict, *args, **kwargs) -> dict:

        # TODO - No need for a loop, need some fancy matrix algebra since they are already tensors :)
        ln = len(batch['image'])
        for idx in range(ln):
            if random.random() > self.probability:
                idx2 = random.randint(0, ln-1)

                # TODO - check to see if we really need to clone... is this really effecting the records object from the dataset class?
                im1 = torch.clone(batch['image'][idx])
                im2 = torch.clone(batch['image'][idx2])

                l1 = torch.clone(batch['label'][idx])
                l2 = torch.clone(batch['label'][idx2])

                alpha = 1.0
                lam = np.random.beta(alpha, alpha)

                batch['image'][idx] = lam * im1 + (1 - lam) * im2
                batch['label'][idx] = lam * l1 + (1 - lam) * l2

        return batch


# TODO - harder to mix up annotations, might need to add enforcers to make sure the mixup is actually useful
class MixUpImageWithAnnotations(Augmentor):

    def handle_transform(self, batch: dict, *args, **kwargs) -> dict:

        # TODO - No need for a loop, need some fancy matrix algebra since they are already tensors :)
        ln = len(batch['image'])
        for idx in range(ln):
            if random.random() > self.probability:
                idx2 = random.randint(0, ln-1)

                # TODO - check to see if we really need to clone... is this really effecting the records object from the dataset class?
                im1 = torch.clone(batch['image'][idx])
                im2 = torch.clone(batch['image'][idx2])

                l1 = batch['annotations'][idx]
                l2 = batch['annotations'][idx2]

                alpha = 1.0
                lam = np.random.beta(alpha, alpha)

                batch['image'][idx] = lam * im1 + (1 - lam) * im2

                # TODO - this assumes that all the bounding boxes are now independent of the classifications present within... maybe this is ok?
                # if an xray has two ailments in the same spot, is that really "50% of x and 50% of y" or two separate boxes with 100% confidence in each?
                # Shouldn't there be weighting based on the beta coefficient? I think we need to do more research here.
                batch['annotations'][idx]['boxes'] = torch.cat([l1['boxes'], l2['boxes']], 0)
                batch['annotations'][idx]['labels'] = torch.cat([l1['labels'], l2['labels']], 0)

        return batch

