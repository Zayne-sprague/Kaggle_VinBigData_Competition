import torch
import numpy as np
import random
from typing import List
import cv2


# https://arxiv.org/pdf/1710.09412.pdf
class MixUpImage:

    def __handle_transform__(self, sample) -> dict:

        # TODO - No need for a loop, need some fancy matrix algebra since they are already tensors :)
        ln = len(sample['image'])
        for idx in range(ln):
            if random.random() > 0.5:
                idx2 = random.randint(0, ln-1)
                im1 = sample['image'][idx]

                im2 = sample['image'][idx2]

                l1 = sample['label'][idx]
                l2 = sample['label'][idx2]

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