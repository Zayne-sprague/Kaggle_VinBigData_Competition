import random
from typing import List, Tuple


class BatchAugmenter:

    def __init__(self):
        self.transforms: List[callable] = []

    def compose(self, *args):
        for arg in args:
            if not isinstance(arg, tuple) and not isinstance(arg, list):
                arg = [arg]
            self.transforms.append(arg)

    def transform(self, batch):
        for transform_pool in self.transforms:
            transformed = False

            while not transformed and len(transform_pool) > 0:
                transform: callable = random.sample(transform_pool, 1)[0]
                batch, transformed = transform(batch)
                transform_pool.remove(transform)

        return batch


class Augmentor:

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, batch: dict, *args, **kwargs) -> Tuple[dict, bool]:
        if random.random() <= self.probability:
            return self.handle_transform(batch, *args, **kwargs), True
        return batch, False

    def handle_transform(self, batch: dict, *args, **kwargs) -> dict:
        raise NotImplementedError()
