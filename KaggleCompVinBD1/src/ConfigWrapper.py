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


    def __hash__(self):
        return f'{self.DEBUG}{self.image_size}{self.include_healthy_annotations}{self.include_records_without_annotations}'
