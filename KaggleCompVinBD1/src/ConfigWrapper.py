import torch

from typing import List

import os
from distutils.util import strtobool


class ConfigWrapper:

    def __init__(self):
        from src import log

        self.DEBUG: bool = strtobool(os.environ.get("DEBUG", False))

        self.use_cache: bool = strtobool(os.environ.get("use_cache", True))

        try:
            self.image_size: int = int(os.environ.get("image_size", 256))
        except ValueError as e:
            log.error("Env variable image_size must be set to an integer")
            raise e
        except Exception as e:
            log.error("Uknown error when loading image_size from environment")
            raise e

        self.include_healthy_annotations: bool = strtobool(os.environ.get("include_healthy_annotations", False))
        self.include_records_without_annotations: bool = strtobool(os.environ.get("include_records_without_annotations", False))

        self.batch_size: int = int(os.environ.get("batch_size", 16))
        self.artificial_batch_size: int = int(os.environ.get("artificial_batch_size", 256))

        if self.batch_size > self.artificial_batch_size:
            log.warn(f"Artificial batch size was smaller than batch size, this is not possible ({self.batch_size} > {self.artificial_batch_size}), artificial batch size set to batch size")
            self.artificial_batch_size = self.batch_size

        self.gpu_count: int = int(os.environ.get("GPU_COUNT", 0))
        self.one_gpu_for_validation: bool = strtobool(os.environ.get("HOLD_ONE_GPU_FOR_VALIDATION", False))
        self.use_gpu: bool = self.gpu_count > 0 and torch.cuda.is_available()

        if self.gpu_count > 0 and not self.use_gpu:
            log.error("Attempted to utilize a GPU but no GPU or CUDA Driver was found.  Defaulting to CPU")
            self.gpu_count = 0

        if self.use_gpu and self.gpu_count > torch.cuda.device_count():
            log.warn(f"Attempted to utilize more GPUs than allowed, setting gpu count to {torch.cuda.device_count()}")
            self.gpu_count = torch.cuda.device_count()

        self.devices: List[torch.device]
        self.validation_device: torch.device
        if self.use_gpu:
            self.devices = [torch.device(f'cuda:{x}') for x in range(self.gpu_count)]
        else:
            self.devices = [torch.device('cpu')]

        if self.use_gpu and self.gpu_count > 1 and self.one_gpu_for_validation:
            self.validation_device = self.devices[-1]
            self.devices = self.devices[:-1]
            self.gpu_count -= 1
        elif self.one_gpu_for_validation and self.gpu_count <= 1:
            log.warn("Attempted to hold off one GPU for validation, but only 1 gpu was found, defaulting validation device to base device")
            self.validation_device = self.devices[0]
        else:
            self.validation_device = self.devices[0]

        self.distribute_across_gpus: bool = strtobool(os.environ.get("distribute_across_gpus", False))

        if self.use_gpu:
            torch.cuda.set_device(0)

    def __hash__(self):
        return f'{self.DEBUG}{self.image_size}{self.include_healthy_annotations}{self.include_records_without_annotations}'
