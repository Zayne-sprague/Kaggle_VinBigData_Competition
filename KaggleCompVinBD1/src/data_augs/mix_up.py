import torch
import numpy as np
import random
from typing import List
import cv2
from shapely.geometry import Polygon

from src import log


# https://arxiv.org/pdf/1710.09412.pdf
class MixUpImage:

    def __handle_transform__(self, sample) -> dict:

        # TODO - No need for a loop, need some fancy matrix algebra since they are already tensors :)
        ln = len(sample['image'])
        for idx in range(ln):
            if random.random() > 0.5:
                idx2 = random.randint(0, ln-1)

                # TODO - check to see if we really need to clone... is this really effecting the records object from the dataset class?
                im1 = torch.clone(sample['image'][idx])
                im2 = torch.clone(sample['image'][idx2])

                l1 = torch.clone(sample['label'][idx])
                l2 = torch.clone(sample['label'][idx2])

                alpha = 1.0
                lam = np.random.beta(alpha, alpha)

                sample['image'][idx] = lam * im1 + (1 - lam) * im2
                sample['label'][idx] = lam * l1 + (1 - lam) * l2

                # im = np.array(torch.unsqueeze(sample['image'][idx], -1).tolist()).astype(np.uint8)
                # im1 = np.array(torch.unsqueeze(im1, -1).tolist()).astype(np.uint8)
                # im2 = np.array(torch.unsqueeze(im2, -1).tolist()).astype(np.uint8)
                #
                # cv2.imshow("im1", im1)
                # cv2.imshow("im2", im2)
                # cv2.imshow(f'im1_{"h" if l1[0] == 1 else "a"} {lam:.2f} | im2_{"h" if l2[0] == 1 else "a"} {1 - lam:.2f}', im)
                # cv2.waitKey(0)

        return sample

    def __call__(self, sample):
        return self.__handle_transform__(sample)


# TODO - harder to mix up annotations, might need to add enforcers to make sure the mixup is actually useful
class MixUpImageWithAnnotations:

    def __handle_transform__(self, sample) -> dict:

        # TODO - No need for a loop, need some fancy matrix algebra since they are already tensors :)
        ln = len(sample['image'])
        for idx in range(ln):
            if random.random() > -0.5:
                idx2 = random.randint(0, ln-1)

                # TODO - check to see if we really need to clone... is this really effecting the records object from the dataset class?
                im1 = torch.clone(sample['image'][idx])
                im2 = torch.clone(sample['image'][idx2])

                l1 = sample['annotations'][idx]
                l2 = sample['annotations'][idx2]

                alpha = 1.0
                lam = np.random.beta(alpha, alpha)

                sample['image'][idx] = lam * im1 + (1 - lam) * im2

                for idx1, (box1, label1) in enumerate(zip(l1['boxes'], l1['labels'])):
                    for _, (box2, label2) in enumerate(zip(l2['boxes'], l2['labels'])):
                        if overlap(box1, box2) or True:
                            sample['annotations'][idx]['labels'][idx1] = lam * label1 + (1 - lam) * label2


        return sample

    def __call__(self, sample):
        return self.__handle_transform__(sample)

def overlap(rect1,rect2) -> bool:
    p1 = Polygon([(rect1[0],rect1[1]), (rect1[1],rect1[1]),(rect1[2],rect1[3]),(rect1[2],rect1[1])])
    p2 = Polygon([(rect2[0],rect2[1]), (rect2[1],rect2[1]),(rect2[2],rect2[3]),(rect2[2],rect2[1])])
    try:
        return(p1.intersects(p2))
    except Exception as e:
        log.critical(f'ERROR IN MIXUP: {e}')
        return False